#ifndef PPCG_OPTIONS_H
#define PPCG_OPTIONS_H

#include <isl/arg.h>

struct ppcg_debug_options {
	int dump_schedule_constraints;
	int dump_schedule;
	int dump_final_schedule;
	int dump_sizes;
	int verbose;
};

struct ppcg_options {
	struct ppcg_debug_options *debug;

	/* Use isl to compute a schedule replacing the original schedule. */
	int reschedule;
	int scale_tile_loops;
	int wrap;

	/* Assume all parameters are non-negative. */
	int non_negative_parameters;
	char *ctx;
	char *sizes;

	int tile_size;

	/* Take advantage of private memory. */
	int use_private_memory;

	/* Take advantage of shared memory. */
	int use_shared_memory;

	/* Maximal amount of shared memory. */
	int max_shared_memory;

	/* The target we generate code for. */
	int target;

	/* Generate OpenMP macros (C target only). */
	int openmp;

	/* Linearize all device arrays. */
	int linearize_device_arrays;

	/* Allow live range to be reordered. */
	int live_range_reordering;

	/* Options to pass to the OpenCL compiler.  */
	char *opencl_compiler_options;
	/* Prefer GPU device over CPU. */
	int opencl_use_gpu;
	/* Number of files to include. */
	int opencl_n_include_file;
	/* Files to include. */
	const char **opencl_include_files;
	/* Print definitions of types in kernels. */
	int opencl_print_kernel_types;
	/* Embed OpenCL kernel code in host code. */
	int opencl_embed_kernel_code;

	/* Name of file for saving isl computed schedule or NULL. */
	char *save_schedule_file;
	/* Name of file for loading schedule or NULL. */
	char *load_schedule_file;
};

ISL_ARG_DECL(ppcg_debug_options, struct ppcg_debug_options,
	ppcg_debug_options_args)
ISL_ARG_DECL(ppcg_options, struct ppcg_options, ppcg_options_args)

#define		PPCG_TARGET_C		0
#define		PPCG_TARGET_CUDA	1
#define		PPCG_TARGET_OPENCL      2

#endif
