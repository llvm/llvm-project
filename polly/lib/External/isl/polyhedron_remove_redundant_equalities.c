/*
 * Copyright 2016      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* This program takes a (possibly parametric) polyhedron as input and
 * prints print a full-dimensional polyhedron with the same number
 * of integer points.
 */

#include <isl/options.h>
#include <isl/printer.h>
#include <isl/set.h>

#include "isl_morph.h"

int main(int argc, char **argv)
{
	isl_ctx *ctx;
	isl_printer *p;
	isl_basic_set *bset;
	isl_morph *morph;
	struct isl_options *options;

	options = isl_options_new_with_defaults();
	argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);
	ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

	bset = isl_basic_set_read_from_file(ctx, stdin);

	morph = isl_basic_set_variable_compression(bset, isl_dim_set);
	bset = isl_morph_basic_set(morph, bset);

	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_print_basic_set(p, bset);
	p = isl_printer_end_line(p);
	isl_printer_free(p);

	isl_basic_set_free(bset);
	isl_ctx_free(ctx);
	return 0;
}
