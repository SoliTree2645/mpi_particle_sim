#include "common.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cassert>
#include <chrono>
// Put any static global variables here that you will use throughout the simulation.

// Domain decomposition and Universal spatial information
static int proc_rows;
static int proc_cols;
static double domain_width;
static double domain_height;

static int proc_row;
static int proc_col;

static double x_min;
static double x_max;
static double y_min;
static double y_max;

// Local spatial cell information
static std::vector<particle_t> local_particles;

// boundary and ghost
enum Direction {TOP = 0, TOP_RIGHT, RIGHT, BOTTOM_RIGHT, BOTTOM, BOTTOM_LEFT, LEFT, TOP_LEFT};
static int neighbors[8];

static std::vector<particle_t> boundary_particles;
static int boundary_counts[8];
static int boundary_offsets[9];

static std::vector<particle_t> ghost_particles;
static int ghost_counts[8];
static int ghost_offsets[9];

static std::vector<particle_t> migrate_send_particles;
static int migrate_send_counts[8];
static int migrate_send_offsets[9]; 

static std::vector<particle_t> migrate_recv_particles;
static int migrate_recv_counts[8];
static int migrate_recv_counts[9];


static std::vector<int> local_cell_counts; // num of particle in each cell
static std::vector<int> local_cell_offsets; // prefix scan of local_cell_counts
static std::vector<int> local_particle_indices; // indices of local particles
static std::vector<int> local_cell_next; // index of putting next particle for each local cell

static int local_cells_per_width;
static int local_cells_per_height;
static double local_cell_width;
static double local_cell_height;

static int local_num_cells;

// Profiling
using Clock = std::chrono::high_resolution_clock;
static double local_put_particle_in_cell_time = 0.0;
static double local_boundary_ghost_time = 0.0;
static double local_force_time = 0.0;
static double ghost_force_time = 0.0;
static double local_move_time = 0.0;
static int step_count;




void apply_force(particle_t& particle, particle_t& neighbor)
{
	double dx = neighbor.x - particle.x;
	double dy = neighbor.y - particle.y;
	double r2 = dx * dx + dy * dy;

	if(r2 > cutoff * cutoff)
		return;

	r2 = fmax(r2, min_r * min_r);
	double r = sqrt(r2);

	double coef = (1 - cutoff / r) / r2 / mass;
	particle.ax += coef * dx;
	particle.ay += coef * dy;
}

void move(particle_t& p, double size)
{
	p.vx += p.ax * dt;
	p.vy += p.ay * dt;
	p.x += p.vx * dt;
	p.y += p.vy * dt;

	while (p.x < 0 || p.x > size)
	{
		p.x = p.x < 0 ? -p.x : 2 * size - p.x;
		p.vx = -p.vx;
	}

	while (p.y < 0 || p.y > size)
	{
		p.y = p.y < 0 ? -p.y : 2 * size - p.y;
		p.vy = -p.vy;
	}
}

int owner_rank(const particle_t& p) 
{
	int col = std::min(static_cast<int>(p.x / domain_width), proc_cols - 1);
	col = std::max(0, col);

	int row = std::min(static_cast<int>(p.y / domain_height), proc_rows - 1);
	row = std::max(0, row);

	int owner = row * proc_cols + col; 

	return owner;
}

int locate_local_cell(double px, double py)
{
	double local_px = px - proc_col * domain_width;
	double local_py = py - proc_row * domain_height;
	int local_col = static_cast<int>(local_px / local_cell_width);
	int local_row = static_cast<int>(local_py / local_cell_height);

	local_col = std::min(local_col, local_cells_per_width - 1);
	local_row = std::min(local_row, local_cells_per_height - 1);

	int cell = local_row * local_cells_per_width + local_col; 
	return cell;
}

void local_put_particle_in_cell()
{
	local_particle_indices.resize(local_particles.size());

	// fill in local_cell_counts
	std::fill(local_cell_counts.begin(), local_cell_counts.end(), 0);
	

	for (int i = 0; i < local_particles.size(); i++)
	{
		int cell = locate_local_cell(local_particles[i].x, local_particles[i].y);

		local_cell_counts[cell] += 1;
	}

	// prefix scan to get local_cell_offsets
	local_cell_offsets[0] = 0; 
	for (int c = 0; c < local_num_cells; c++)
	{
		local_cell_offsets[c + 1] = local_cell_offsets[c] + local_cell_counts[c];
	}

	// local_cell_next is same as local_cell_offsets initially
	for (int c = 0; c < local_num_cells; c++)
	{
		local_cell_next[c] = local_cell_offsets[c]; 
	}

	// put particle indices in local cell
	for (int i = 0; i < local_particles.size(); i++)
	{
		int cell = locate_local_cell(local_particles[i].x, local_particles[i].y);

		local_particle_indices[local_cell_next[cell]] = i;

		local_cell_next[cell] += 1;
		
	}
}

void neighbors_setup(int rank)
{
	if(proc_row == proc_rows - 1)
	{
		neighbors[TOP] = MPI_PROC_NULL;
	}else
	{
		neighbors[TOP] = rank + proc_cols;
	}
	
	
	if(proc_col == proc_cols - 1)
	{
		neighbors[RIGHT] = MPI_PROC_NULL;
	}else
	{
		neighbors[RIGHT] = rank + 1;
	}

	// check if bottom is boundary
	if(proc_row == 0)
	{
		neighbors[BOTTOM] = MPI_PROC_NULL;
	}else
	{
		neighbors[BOTTOM] = rank - proc_cols;
	}

	// check if left is boundary
	if(proc_col == 0)
	{
		neighbors[LEFT] = MPI_PROC_NULL;
	}else
	{
		neighbors[LEFT] = rank - 1;
	}

	if(proc_row == proc_rows - 1 || proc_col == proc_cols - 1)
	{
		neighbors[TOP_RIGHT] = MPI_PROC_NULL;
	}else
	{
		neighbors[TOP_RIGHT] = neighbors[TOP] + 1;
	}

	if(proc_row == 0 || proc_col == proc_cols - 1)
	{
		neighbors[BOTTOM_RIGHT] = MPI_PROC_NULL;
	}else
	{
		neighbors[BOTTOM_RIGHT] = neighbors[BOTTOM] + 1;
	}

	if(proc_row == 0 || proc_col == 0)
	{
		neighbors[BOTTOM_LEFT] = MPI_PROC_NULL;
	}else
	{
		neighbors[BOTTOM_LEFT] = neighbors[BOTTOM] - 1;
	}

	if(proc_row == proc_rows - 1 || proc_col == 0)
	{
		neighbors[TOP_LEFT] = MPI_PROC_NULL;
	}else
	{
		neighbors[TOP_LEFT] = neighbors[TOP] - 1;
	}

}


void init_simulation(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
	// You can use this space to initialize data objects that you may need
	// This function will be called once before the algorithm begins
	// Do not do any particle simulation here
	
	
	// Domain decomposition
	int dims[2] = {0, 0};
	MPI_Dims_create(num_procs, 2, dims);

	proc_rows = dims[0];
	proc_cols = dims[1];
	domain_width = size / proc_cols;
	domain_height = size / proc_rows;

	proc_row = rank / proc_cols;
	proc_col = rank % proc_cols; 

	x_min = domain_width * proc_col;
	x_max = domain_width * (proc_col + 1);
	y_min = domain_height * proc_row;
	y_max = domain_height * (proc_row + 1);
	
	
	neighbors_setup(rank);
	


	for (int i = 0; i < num_parts; i++)
	{
		 
		if(owner_rank(parts[i]) == rank)
		{
			local_particles.push_back(parts[i]);
			
		}
	}
	


	int local_num_parts = static_cast<int>(local_particles.size());
	int total_count = 0;

	MPI_Reduce(&local_num_parts, &total_count, 1, 
			MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

	if(rank == 0)
	{
		std::cout << "Domain decomposition information. " << std::endl;
		std::cout << "Num_procs: " << num_procs << ", proc_rows: " << proc_rows << ", proc_cols: " << proc_cols << std::endl;
		std::cout << "Whole space size: " << size << ". domain width: " << domain_width << ", domain height: " << domain_height << std::endl; 
		std::cout << "Total owned particles: " << total_count << "/ " << num_parts << std::endl;
	}

	//Initialize the CSR local spatial cell representation
    local_cells_per_width = std::max(1, static_cast<int>(domain_width / cutoff));
	local_cells_per_height = std::max(1, static_cast<int>(domain_height / cutoff));
	local_cell_width = domain_width / local_cells_per_width;
	local_cell_height = domain_height / local_cells_per_height;

	local_num_cells = local_cells_per_width * local_cells_per_height;

	local_cell_counts.resize(local_num_cells);
	local_cell_offsets.resize(local_num_cells + 1);
	local_cell_next.resize(local_num_cells);
	local_particle_indices.resize(local_particles.size()); 




}

void local_identify_boundary_cells()
{
	boundary_particles.clear();
	// top
	int top_bd_count = 0;
	if(neighbors[TOP] != MPI_PROC_NULL)
	{
		for(int i = 0; i < local_cells_per_width; i++)
		{
			int cell = local_cells_per_width * (local_cells_per_height - 1) + i;
			int begin = local_cell_offsets[cell];
			int end = local_cell_offsets[cell + 1];
			for(int k = begin; k < end; k++)
			{
				int j = local_particle_indices[k];
				boundary_particles.push_back(local_particles[j]);
				top_bd_count += 1; 
				
			}

		}
	}

	int top_right_bd_count = 0;
	if(neighbors[TOP_RIGHT] != MPI_PROC_NULL)
	{
		int cell = local_num_cells - 1;
		int begin = local_cell_offsets[cell];
		int end = local_cell_offsets[cell + 1];
		for(int k = begin; k < end; k++)
		{
			int j = local_particle_indices[k];
			boundary_particles.push_back(local_particles[j]);
			top_right_bd_count += 1;
		}
	}

	// right
	int right_bd_count = 0;
	if(neighbors[RIGHT] != MPI_PROC_NULL)
	{
		for(int i = 0; i < local_cells_per_height; i++)
		{
			int cell = i * local_cells_per_width + (local_cells_per_width - 1);
			int begin = local_cell_offsets[cell];
			int end = local_cell_offsets[cell + 1];
			for(int k = begin; k < end; k++)
			{
				int j = local_particle_indices[k];
				boundary_particles.push_back(local_particles[j]);
				right_bd_count += 1;
			}
		}
	}

	int bottom_right_bd_count = 0;
	if(neighbors[BOTTOM_RIGHT] != MPI_PROC_NULL)
	{
		int cell = local_cells_per_width - 1;
		int begin = local_cell_offsets[cell];
		int end = local_cell_offsets[cell + 1];
		for(int k = begin; k < end; k++)
		{
			int j = local_particle_indices[k];
			boundary_particles.push_back(local_particles[j]);
			bottom_right_bd_count += 1;
		}
	}


	// bottom
	int bottom_bd_count = 0;
	if(neighbors[BOTTOM] != MPI_PROC_NULL)
	{
		for(int cell = 0; cell < local_cells_per_width; cell++)
		{
			int begin = local_cell_offsets[cell];
			int end = local_cell_offsets[cell + 1];
			for(int k = begin; k < end; k++)
			{
				int j = local_particle_indices[k];
				boundary_particles.push_back(local_particles[j]);
				bottom_bd_count += 1;
			}
		}
	}

	int bottom_left_bd_count = 0;
	if(neighbors[BOTTOM_LEFT] != MPI_PROC_NULL)
	{
		int begin = local_cell_offsets[0];
		int end = local_cell_offsets[1];
		for(int k = begin; k < end; k++)
		{
			int j = local_particle_indices[k];
			boundary_particles.push_back(local_particles[j]);
			bottom_left_bd_count += 1;
		}
	}

	// left
	int left_bd_count = 0;
	if (neighbors[LEFT] != MPI_PROC_NULL)
	{
		for(int i = 0; i < local_cells_per_height; i++)
		{
			int cell = i * local_cells_per_width;
			int begin = local_cell_offsets[cell];
			int end = local_cell_offsets[cell + 1];
			for(int k = begin; k < end; k++)
			{
				int j = local_particle_indices[k];
				boundary_particles.push_back(local_particles[j]);
				left_bd_count += 1;
			}

		}
	}

	int top_left_bd_count = 0;
	if(neighbors[TOP_LEFT] != MPI_PROC_NULL)
	{
		int cell = local_num_cells - local_cells_per_width;
		int begin = local_cell_offsets[cell];
		int end = local_cell_offsets[cell + 1];
		for(int k = begin; k < end; k++)
		{
			int j = local_particle_indices[k];
			boundary_particles.push_back(local_particles[j]);
			top_left_bd_count += 1;
		}
	}


	boundary_counts[TOP] = top_bd_count;
	boundary_counts[TOP_RIGHT] = top_right_bd_count;
	boundary_counts[RIGHT] = right_bd_count;
	boundary_counts[BOTTOM_RIGHT] = bottom_right_bd_count;
	boundary_counts[BOTTOM] = bottom_bd_count;
	boundary_counts[BOTTOM_LEFT] = bottom_left_bd_count;
	boundary_counts[LEFT] = left_bd_count;
	boundary_counts[TOP_LEFT] = top_left_bd_count;
	
	
	
	boundary_offsets[0] = 0;
	for (int i = 0; i < 8; i++)
	{
		boundary_offsets[i + 1] = boundary_offsets[i] + boundary_counts[i]; 
	}
	
	
}

int find_neighbor_direction(int dst)
{
	for(int d = 0; d < 8; d++)
	{
		if(dst == neighbors[d])
		{
			return d;
		}
	}

	return -1;
}

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {	
    auto t0 = Clock::now();
    local_put_particle_in_cell();
	

    auto t1 = Clock::now();

    // identity the boundary particles after putting particles in cell
	local_identify_boundary_cells();
	assert(boundary_offsets[8] == static_cast<int>(boundary_particles.size()));
	for(int d = 0; d < 8; d++)
	{
		assert(boundary_offsets[d + 1] - boundary_offsets[d] == boundary_counts[d]);
	}

	// exchange boundary and find ghosts
	std::fill(ghost_counts, ghost_counts + 8, 0);
	const int GHOST_COUNT_TAG = 100;
	MPI_Request count_requests[16];
	int cnt_req_num = 0;

	for(int d = 0; d < 8; d++)
	{
		if(neighbors[d] != MPI_PROC_NULL)
		{
			MPI_Irecv(
				&ghost_counts[d], 
				1, 
				MPI_INT, 
				neighbors[d], 
				GHOST_COUNT_TAG,
				MPI_COMM_WORLD,
				&count_requests[cnt_req_num++]
			);
		}
	}

	for(int d = 0; d < 8; d++)
	{
		if(neighbors[d] != MPI_PROC_NULL)
		{
			MPI_Isend(&boundary_counts[d], 
				1, 
				MPI_INT, 
				neighbors[d], 
				GHOST_COUNT_TAG, 
				MPI_COMM_WORLD, 
				&count_requests[cnt_req_num++]
			);
		}
	}
	MPI_Waitall(cnt_req_num, count_requests, MPI_STATUSES_IGNORE);

	ghost_offsets[0] = 0;
	for(int i = 0; i < 8; i++)
	{
		ghost_offsets[i + 1] = ghost_offsets[i] + ghost_counts[i];
	}
	ghost_particles.resize(ghost_offsets[8]);
	assert(ghost_offsets[8] == static_cast<int>(ghost_particles.size()));

	const int GHOST_DATA_TAG = 200;
	MPI_Request data_requests[16];
	int data_req_num = 0;

	for(int d = 0; d < 8; d++)
	{
		if(neighbors[d] != MPI_PROC_NULL)
		{
			particle_t* recv_ptr = ghost_counts[d] > 0 ? ghost_particles.data() + ghost_offsets[d] : nullptr;
			MPI_Irecv(recv_ptr, 
				ghost_counts[d], 
				PARTICLE, 
				neighbors[d], 
				GHOST_DATA_TAG, 
				MPI_COMM_WORLD, 
				&data_requests[data_req_num++]
			);
		}
	}
	
	for(int d = 0; d < 8; d++)
	{
		if(neighbors[d] != MPI_PROC_NULL)
		{
			particle_t* send_ptr = boundary_counts[d] > 0 ? boundary_particles.data() + boundary_offsets[d] : nullptr;
			MPI_Isend(send_ptr, 
				boundary_counts[d], 
				PARTICLE, 
				neighbors[d], 
				GHOST_DATA_TAG, 
				MPI_COMM_WORLD, 
				&data_requests[data_req_num++]
			);
		}
	}

	MPI_Waitall(data_req_num, data_requests, MPI_STATUSES_IGNORE);

	auto t2 = Clock::now();


    // apply local force
    
    for (int i = 0; i < local_particles.size(); i++)
    {
		local_particles[i].ax = 0;
		local_particles[i].ay = 0;

		int cell = locate_local_cell(local_particles[i].x, local_particles[i].y);
		int cell_col = cell % local_cells_per_width;
		int cell_row = cell / local_cells_per_width;
		
		for (int da = -1; da <= 1; da++)
		{
			for (int db = -1; db <= 1; db++)
			{
				int neighbor_cell_col = cell_col + da;
				int neighbor_cell_row = cell_row + db;

				if(neighbor_cell_col < 0 || neighbor_cell_col >= local_cells_per_width ||
						neighbor_cell_row < 0 || neighbor_cell_row >= local_cells_per_height)
				{
					continue;
				}
				int neighbor_cell = neighbor_cell_col + neighbor_cell_row * local_cells_per_width;

				int begin = local_cell_offsets[neighbor_cell];
				int end = local_cell_offsets[neighbor_cell + 1];

				for (int k = begin; k < end; k++)
				{
					int j = local_particle_indices[k];
					apply_force(local_particles[i], local_particles[j]);
				}
			}
		}
    }

	auto t3 = Clock::now();


	// apply ghost force
	for (int i = 0; i < local_particles.size(); i++)
	{
		for(int g = 0; g < static_cast<int>(ghost_particles.size()); g++)
		{
			apply_force(local_particles[i], ghost_particles[g]);
		}
	}

	auto t4 = Clock::now();
    
    
    //Move particles
	migrate_send_particles.clear();
	migrate_send_counts[8] = {0};
    for(int i = 0; i < local_particles.size(); i++)
    {
	    move(local_particles[i], size);

		int dst_rank = owner_rank(local_particles[i]);
		if(dst_rank != rank)
		{
			int direction = find_neighbor_direction(dst_rank);
			assert(direction != -1);
			migrate_send_counts[direction] += 1;
		}
	
    }

	migrate_send_offsets[0] = 0;
	for (int i = 0; i < 8; i++)
	{
		migrate_send_offsets[i + 1] = migrate_send_offsets[i] + migrate_send_counts[i];
	}
	migrate_send_particles.resize(migrate_send_offsets[8]);


	std::vector<particle_t> staying_particles;
	staying_particles.reserve(local_particles.size());

	int migrate_send_dir_next[8];
	for(int i = 0; i < 8; i++)
	{
		migrate_send_dir_next[i] = migrate_send_offsets[i];
	}

	for(const particle_t& p: local_particles)
	{
		int dst = owner_rank(p);
		if(dst == rank)
		{
			staying_particles.push_back(p);
		}else
		{
			int direction = find_neighbor_direction(dst);
			migrate_send_particles[migrate_send_dir_next[direction]++] = p;
		}
	}



    auto end = Clock::now();
    step_count += 1;

    local_put_particle_in_cell_time += std::chrono::duration<double>(t1 - t0).count();
	local_boundary_ghost_time += std::chrono::duration<double>(t2 - t1).count();
    local_force_time += std::chrono::duration<double>(t3 - t2).count();
	ghost_force_time += std::chrono::duration<double>(t4 - t3).count();
    local_move_time += std::chrono::duration<double>(end - t4).count();

    if(rank == 0)
    {
	    if(step_count == nsteps)
	    {
		    std::cout << "Put particle in local cell time: " << local_put_particle_in_cell_time << std::endl;
			std::cout << "Local boundary and ghost communication time: " << local_boundary_ghost_time << std::endl;
		    std::cout << "Local force time: " << local_force_time << std::endl;
			std::cout << "Ghost force time: " << ghost_force_time << std::endl;
		    std::cout << "local move time: " << local_move_time << std::endl;
	    }
    }

    
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    //
    int local_num_parts = static_cast<int>(local_particles.size());
    int recvcounts[num_procs];

    MPI_Gather(
	        &local_num_parts, 
		    1, 
		    MPI_INT, 
		    rank == 0 ? &recvcounts[0] : nullptr, 
		    1, 
		    MPI_INT, 
		    0, 
		    MPI_COMM_WORLD
		    ); 

    int displs[num_procs];

    if(rank == 0)
    {
    	displs[0] = 0;

    	for (int i = 0; i < num_procs - 1; i++)
    	{
	    	displs[i + 1] = displs[i] + recvcounts[i];
	
    	}
    }


    MPI_Gatherv(
		    local_particles.data(), 
		    local_num_parts, 
		    PARTICLE, 
		    parts, 
		    recvcounts, 
		    displs, 
		    PARTICLE, 
		    0, 
		    MPI_COMM_WORLD
		    );

    if (rank == 0)
    {
    	std::sort(parts, parts + num_parts, 
	      	[](const particle_t& a, const particle_t& b)
	      	{
	      		return a.id < b.id;
	      	}
		);

    }


}
