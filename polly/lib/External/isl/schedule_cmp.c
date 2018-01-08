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
#include <isl/schedule.h>

struct options {
	struct isl_options *isl;
	char *schedule1;
	char *schedule2;
};

ISL_ARGS_START(struct options, options_args)
ISL_ARG_CHILD(struct options, isl, "isl", &isl_options_args, "isl options")
ISL_ARG_ARG(struct options, schedule1, "schedule1", NULL)
ISL_ARG_ARG(struct options, schedule2, "schedule2", NULL)
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

/* Given two YAML descriptions of isl_schedule objects, check whether
 * they are equivalent.
 * Return EXIT_SUCCESS if they are and EXIT_FAILURE if they are not
 * or if anything else went wrong.
 */
int main(int argc, char **argv)
{
	isl_ctx *ctx;
	struct options *options;
	FILE *input1, *input2;
	isl_bool equal;
	isl_schedule *s1, *s2;

	options = options_new_with_defaults();
	if (!options)
		return EXIT_FAILURE;

	ctx = isl_ctx_alloc_with_options(&options_args, options);
	argc = options_parse(options, argc, argv, ISL_ARG_ALL);

	input1 = open_or_die(options->schedule1);
	input2 = open_or_die(options->schedule2);
	s1 = isl_schedule_read_from_file(ctx, input1);
	s2 = isl_schedule_read_from_file(ctx, input2);

	equal = isl_schedule_plain_is_equal(s1, s2);
	if (equal < 0)
		return EXIT_FAILURE;
	if (!equal)
		die("schedules differ");

	isl_schedule_free(s1);
	isl_schedule_free(s2);
	fclose(input1);
	fclose(input2);
	isl_ctx_free(ctx);

	return EXIT_SUCCESS;
}
