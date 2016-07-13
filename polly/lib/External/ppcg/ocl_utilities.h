#ifndef OCL_UTILITIES_H
#define OCL_UTILITIES_H

#if defined(__APPLE__)
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

/* Return the OpenCL error string for a given error number.
 */
const char *opencl_error_string(cl_int error);

/* Find a GPU or a CPU associated with the first available platform.
 * If use_gpu is set, then this function first tries to look for a GPU
 * in the first available platform.
 * If this fails or if use_gpu is not set, then it tries to use the CPU.
 */
cl_device_id opencl_create_device(int use_gpu);

/* Create an OpenCL program from a string and compile it.
 */
cl_program opencl_build_program_from_string(cl_context ctx, cl_device_id dev,
	const char *program_source, size_t program_size,
	const char *opencl_options);

/* Create an OpenCL program from a source file and compile it.
 */
cl_program opencl_build_program_from_file(cl_context ctx, cl_device_id dev,
	const char* filename, const char* opencl_options);

#endif
