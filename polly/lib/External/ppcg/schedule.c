/*
 * Copyright 2010-2011 INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <string.h>

#include <isl/set.h>
#include <isl/map.h>
#include <isl/constraint.h>

#include "schedule.h"

/* Add parameters with identifiers "ids" to "set".
 */
static __isl_give isl_set *add_params(__isl_take isl_set *set,
	__isl_keep isl_id_list *ids)
{
	int i, n;
	unsigned nparam;

	n = isl_id_list_n_id(ids);

	nparam = isl_set_dim(set, isl_dim_param);
	set = isl_set_add_dims(set, isl_dim_param, n);

	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = isl_id_list_get_id(ids, i);
		set = isl_set_set_dim_id(set, isl_dim_param, nparam + i, id);
	}

	return set;
}

/* Equate the dimensions of "set" starting at "first" to
 * freshly created parameters with identifiers "ids".
 * The number of equated dimensions is equal to the number of elements in "ids".
 */
static __isl_give isl_set *parametrize(__isl_take isl_set *set,
	int first, __isl_keep isl_id_list *ids)
{
	int i, n;
	unsigned nparam;

	nparam = isl_set_dim(set, isl_dim_param);

	set = add_params(set, ids);

	n = isl_id_list_n_id(ids);
	for (i = 0; i < n; ++i)
		set = isl_set_equate(set, isl_dim_param, nparam + i,
					isl_dim_set, first + i);

	return set;
}

/* Given a parameter space "space", create a set of dimension "len"
 * of which the dimensions starting at "first" are equated to
 * freshly created parameters with identifiers "ids".
 */
__isl_give isl_set *parametrization(__isl_take isl_space *space,
	int len, int first, __isl_keep isl_id_list *ids)
{
	isl_set *set;

	space = isl_space_set_from_params(space);
	space = isl_space_add_dims(space, isl_dim_set, len);
	set = isl_set_universe(space);

	return parametrize(set, first, ids);
}

/* Load and return a schedule from a file called "filename".
 */
static __isl_give isl_schedule *load_schedule(isl_ctx *ctx,
	const char *filename)
{
	FILE *file;
	isl_schedule *schedule;

	file = fopen(filename, "r");
	if (!file) {
		fprintf(stderr, "Unable to open '%s' for reading\n", filename);
		return NULL;
	}
	schedule = isl_schedule_read_from_file(ctx, file);
	fclose(file);

	return schedule;
}

/* Save the schedule "schedule" to a file called "filename".
 * The schedule is printed in block style.
 */
static void save_schedule(__isl_keep isl_schedule *schedule,
	const char *filename)
{
	FILE *file;
	isl_ctx *ctx;
	isl_printer *p;

	if (!schedule)
		return;

	file = fopen(filename, "w");
	if (!file) {
		fprintf(stderr, "Unable to open '%s' for writing\n", filename);
		return;
	}
	ctx = isl_schedule_get_ctx(schedule);
	p = isl_printer_to_file(ctx, file);
	p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
	p = isl_printer_print_schedule(p, schedule);
	isl_printer_free(p);
	fclose(file);
}

/* Obtain a schedule, either by reading it form a file
 * or by computing it using "compute".
 * Also take care of saving the computed schedule and/or
 * dumping the obtained schedule if requested by the user.
 */
__isl_give isl_schedule *ppcg_get_schedule(isl_ctx *ctx,
	struct ppcg_options *options,
	__isl_give isl_schedule *(*compute)(void *user), void *user)
{
	isl_schedule *schedule;

	if (options->load_schedule_file) {
		schedule = load_schedule(ctx, options->load_schedule_file);
	} else {
		schedule = compute(user);
		if (options->save_schedule_file)
			save_schedule(schedule, options->save_schedule_file);
	}
	if (options->debug->dump_schedule)
		isl_schedule_dump(schedule);

	return schedule;
}

/* Mark all dimensions in the band node "node" to be of "type".
 */
__isl_give isl_schedule_node *ppcg_set_schedule_node_type(
	__isl_take isl_schedule_node *node, enum isl_ast_loop_type type)
{
	int i, n;

	n = isl_schedule_node_band_n_member(node);
	for (i = 0; i < n; ++i)
		node = isl_schedule_node_band_member_set_ast_loop_type(node, i,
							type);

	return node;
}
