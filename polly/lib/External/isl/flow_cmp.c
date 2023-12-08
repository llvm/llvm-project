/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include <stdlib.h>

#include <isl/arg.h>
#include <isl/options.h>
#include <isl/union_map.h>
#include <isl/stream.h>

struct options {
	struct isl_options *isl;
	char *flow1;
	char *flow2;
};

ISL_ARGS_START(struct options, options_args)
ISL_ARG_CHILD(struct options, isl, "isl", &isl_options_args, "isl options")
ISL_ARG_ARG(struct options, flow1, "flow1", NULL)
ISL_ARG_ARG(struct options, flow2, "flow2", NULL)
ISL_ARGS_END

ISL_ARG_DEF(options, struct options, options_args)

static void die(const char *msg)
{
	fprintf(stderr, "%s\n", msg);
	exit(EXIT_FAILURE);
}

static FILE *open_or_die(const char *filename)
{
	FILE *file;

	file = fopen(filename, "r");
	if (!file) {
		fprintf(stderr, "Unable to open %s\n", filename);
		exit(EXIT_FAILURE);
	}
	return file;
}

#undef BASE
#define BASE union_map
#include "read_in_string_templ.c"

/* Given two YAML descriptions of isl_union_flow objects, check whether
 * they are equivalent.
 * Return EXIT_SUCCESS if they are and EXIT_FAILURE if they are not
 * or if anything else went wrong.
 *
 * The descriptions are checked field by field, meaning that the fields
 * are expected to appear in the same order in both inputs.
 */
int main(int argc, char **argv)
{
	isl_bool more;
	isl_ctx *ctx;
	struct options *options;
	FILE *input1, *input2;
	isl_stream *s1, *s2;

	options = options_new_with_defaults();
	if (!options)
		return EXIT_FAILURE;

	ctx = isl_ctx_alloc_with_options(&options_args, options);
	argc = options_parse(options, argc, argv, ISL_ARG_ALL);

	input1 = open_or_die(options->flow1);
	input2 = open_or_die(options->flow2);
	s1 = isl_stream_new_file(ctx, input1);
	s2 = isl_stream_new_file(ctx, input2);

	if (isl_stream_yaml_read_start_mapping(s1) < 0)
		isl_die(ctx, isl_error_unknown, "arg1 not a YAML mapping",
			return EXIT_FAILURE);
	if (isl_stream_yaml_read_start_mapping(s2) < 0)
		isl_die(ctx, isl_error_unknown, "arg2 not a YAML mapping",
			return EXIT_FAILURE);

	while ((more = isl_stream_yaml_next(s1)) == isl_bool_true) {
		isl_bool more2;
		isl_bool equal;
		isl_union_map *umap1, *umap2;

		more2 = isl_stream_yaml_next(s2);
		if (more2 < 0)
			return EXIT_FAILURE;
		if (!more2)
			isl_die(ctx, isl_error_unknown, "arg2 shorter",
				return EXIT_FAILURE);
		if (isl_stream_eat(s1, ISL_TOKEN_IDENT) < 0)
			return EXIT_FAILURE;
		if (isl_stream_eat(s2, ISL_TOKEN_IDENT) < 0)
			return EXIT_FAILURE;
		more = isl_stream_yaml_next(s1);
		more2 = isl_stream_yaml_next(s2);
		if (more < 0 || more2 < 0)
			return EXIT_FAILURE;
		if (!more || !more2)
			isl_die(ctx, isl_error_unknown, "missing value",
				return EXIT_FAILURE);

		umap1 = read_union_map(s1);
		umap2 = read_union_map(s2);
		equal = isl_union_map_is_equal(umap1, umap2);
		isl_union_map_free(umap1);
		isl_union_map_free(umap2);
		if (equal < 0)
			return EXIT_FAILURE;
		if (!equal)
			die("field not equal");
	}
	if (more < 0)
		return EXIT_FAILURE;


	if (isl_stream_yaml_read_end_mapping(s1) < 0)
		return EXIT_FAILURE;
	if (isl_stream_yaml_read_end_mapping(s2) < 0)
		return EXIT_FAILURE;

	isl_stream_free(s1);
	isl_stream_free(s2);
	fclose(input1);
	fclose(input2);
	isl_ctx_free(ctx);

	return EXIT_SUCCESS;
}
