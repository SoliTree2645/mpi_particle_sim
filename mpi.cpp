#include "common.h"
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
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

static std::vector<int> local_cell_counts; // num of particle in each cell
static std::vector<int> local_cell_offsets; // prefix scan of local_cell_counts
static std::vector<int> local_particle_indices; // indices of local particles
static std::vector<int> local_cell_next; // index of putting next particle for each local cell

static int local_cells_per_width;
static int local_cells_per_height;
static double local_cell_width;
static double local_cell_height;

int local_num_cells;



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

int locate_cell(double px, double py)
{
	int local_col = static_cast<int>(px / local_cell_width);
	int local_row = static_cast<int>(py / local_cell_height);

	local_col = std::min(local_col, local_cells_per_width - 1);
	local_row = std::min(local_row, local_cells_per_height - 1);

	int cell = local_row * local_cells_per_width + local_col; 
	return cell;
}

void local_put_particle_in_cell()
{
	// fill in local_cell_counts
	std::fill(local_cell_counts.begin(), local_cell_counts.end(), 0);
	int local_num_parts = local_particles.size();

	for (int i = 0; i < local_num_parts; i++)
	{
		int cell = locate_cell(local_particles[i].x, local_particles[i].y);

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
	for (int i = 0; i < local_num_parts; i++)
	{
		int cell = locate_cell(local_particles[i].x, local_particles[i].y);

		local_particle_indices[local_cell_next[cell]] = i;

		local_cell_next[cell] += 1;
		
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


	for (int i = 0; i < num_parts; i++)
	{
		 
		if(owner_rank(parts[i]) == rank)
		{
			local_particles.push_back(parts[i]);
			
		}
	}

	int local_count = static_cast<int>(local_particles.size());
	int total_count = 0;

	MPI_Reduce(&local_count, &total_count, 1, 
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

void simulate_one_step(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    local_put_particle_in_cell();


    // apply force
    for (int i = 0; i < local_particles.size(); i++)
    {
	local_particles[i].ax = 0;
	local_particles[i].ay = 0;

	int cell_col = static_cast<int>(local_particles[i].x / local_cell_width);
	int cell_row = static_cast<int>(local_particles[i].y / local_cell_height);

	cell_col = std::min(cell_col, local_cells_per_width - 1);
	cell_row = std::min(cell_row, local_cells_per_height - 1);

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
    //Move particles
    for(int i = 0; i < local_particles.size(); i++)
    {
	    move(local_particles[i], size);
	
    }
}

void gather_for_save(particle_t* parts, int num_parts, double size, int rank, int num_procs) {
    // Write this function such that at the end of it, the master (rank == 0)
    // processor has an in-order view of all particles. That is, the array
    // parts is complete and sorted by particle id.
    //
    for (int i = 0; i < local_particles.size(); i++)
    {
	    parts[i] = local_particles[i];
    }
}
