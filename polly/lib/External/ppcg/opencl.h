#ifndef _OPENCL_H
#define _OPENCL_H

#include <pet.h>
#include "ppcg_options.h"
#include "ppcg.h"

int generate_opencl(isl_ctx *ctx, struct ppcg_options *options,
	const char *input, const char *output);

#endif
