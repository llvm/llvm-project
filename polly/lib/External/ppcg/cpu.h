#ifndef _CPU_H
#define _CPU_H

#include <isl/ctx.h>

#include "ppcg.h"

struct ppcg_options;

__isl_give isl_printer *print_cpu(__isl_take isl_printer *p,
	struct ppcg_scop *ps, struct ppcg_options *options);
int generate_cpu(isl_ctx *ctx, struct ppcg_options *options,
	const char *input, const char *output);

#endif
