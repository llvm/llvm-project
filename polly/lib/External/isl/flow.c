/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

/* This program takes an isl_union_access_info object as input and
 * prints the corresponding dependences.
 */

#include <isl/options.h>
#include <isl/printer.h>
#include <isl/union_map.h>
#include <isl/flow.h>
#include <isl/schedule.h>

int main(int argc, char **argv)
{
	isl_ctx *ctx;
	isl_printer *p;
	isl_union_access_info *access;
	isl_union_flow *flow;
	struct isl_options *options;

	options = isl_options_new_with_defaults();
	argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);
	ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

	access = isl_union_access_info_read_from_file(ctx, stdin);
	flow = isl_union_access_info_compute_flow(access);

	p = isl_printer_to_file(ctx, stdout);
	p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
	p = isl_printer_print_union_flow(p, flow);
	isl_printer_free(p);

	isl_union_flow_free(flow);

	isl_ctx_free(ctx);

	return 0;
}
