/*
 * Copyright 2012 INRIA Paris-Rocquencourt
 * Copyright 2012 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Tobias Grosser, INRIA Paris-Rocquencourt,
 * Domaine de Voluceau, Rocquenqourt, B.P. 105,
 * 78153 Le Chesnay Cedex France
 * and Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/flow.h>
#include <isl/map.h>
#include <isl/ast_build.h>
#include <isl/schedule.h>
#include <isl/schedule_node.h>
#include <pet.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "cpu.h"
#include "print.h"
#include "schedule.h"
#include "util.h"

/* Representation of a statement inside a generated AST.
 *
 * "stmt" refers to the original statement.
 * "ref2expr" maps the reference identifier of each access in
 * the statement to an AST expression that should be printed
 * at the place of the access.
 */
struct ppcg_stmt {
	struct pet_stmt *stmt;

	isl_id_to_ast_expr *ref2expr;
};

static void ppcg_stmt_free(void *user)
{
	struct ppcg_stmt *stmt = user;

	if (!stmt)
		return;

	isl_id_to_ast_expr_free(stmt->ref2expr);

	free(stmt);
}

/* Derive the output file name from the input file name.
 * 'input' is the entire path of the input file. The output
 * is the file name plus the additional extension.
 *
 * We will basically replace everything after the last point
 * with '.ppcg.c'. This means file.c becomes file.ppcg.c
 */
static FILE *get_output_file(const char *input, const char *output)
{
	char name[PATH_MAX];
	const char *ext;
	const char ppcg_marker[] = ".ppcg";
	int len;
	FILE *file;

	len = ppcg_extract_base_name(name, input);

	strcpy(name + len, ppcg_marker);
	ext = strrchr(input, '.');
	strcpy(name + len + sizeof(ppcg_marker) - 1, ext ? ext : ".c");

	if (!output)
		output = name;

	file = fopen(output, "w");
	if (!file) {
		fprintf(stderr, "Unable to open '%s' for writing\n", output);
		return NULL;
	}

	return file;
}

/* Data used to annotate for nodes in the ast.
 */
struct ast_node_userinfo {
	/* The for node is an openmp parallel for node. */
	int is_openmp;
};

/* Information used while building the ast.
 */
struct ast_build_userinfo {
	/* The current ppcg scop. */
	struct ppcg_scop *scop;

	/* Are we currently in a parallel for loop? */
	int in_parallel_for;
};

/* Check if the current scheduling dimension is parallel.
 *
 * We check for parallelism by verifying that the loop does not carry any
 * dependences.
 * If the live_range_reordering option is set, then this currently
 * includes the order dependences.  In principle, non-zero order dependences
 * could be allowed, but this would require privatization and/or expansion.
 *
 * Parallelism test: if the distance is zero in all outer dimensions, then it
 * has to be zero in the current dimension as well.
 * Implementation: first, translate dependences into time space, then force
 * outer dimensions to be equal.  If the distance is zero in the current
 * dimension, then the loop is parallel.
 * The distance is zero in the current dimension if it is a subset of a map
 * with equal values for the current dimension.
 */
static int ast_schedule_dim_is_parallel(__isl_keep isl_ast_build *build,
	struct ppcg_scop *scop)
{
	isl_union_map *schedule, *deps;
	isl_map *schedule_deps, *test;
	isl_space *schedule_space;
	unsigned i, dimension, is_parallel;

	schedule = isl_ast_build_get_schedule(build);
	schedule_space = isl_ast_build_get_schedule_space(build);

	dimension = isl_space_dim(schedule_space, isl_dim_out) - 1;

	deps = isl_union_map_copy(scop->dep_flow);
	deps = isl_union_map_union(deps, isl_union_map_copy(scop->dep_false));
	if (scop->options->live_range_reordering) {
		isl_union_map *order = isl_union_map_copy(scop->dep_order);
		deps = isl_union_map_union(deps, order);
	}
	deps = isl_union_map_apply_range(deps, isl_union_map_copy(schedule));
	deps = isl_union_map_apply_domain(deps, schedule);

	if (isl_union_map_is_empty(deps)) {
		isl_union_map_free(deps);
		isl_space_free(schedule_space);
		return 1;
	}

	schedule_deps = isl_map_from_union_map(deps);

	for (i = 0; i < dimension; i++)
		schedule_deps = isl_map_equate(schedule_deps, isl_dim_out, i,
					       isl_dim_in, i);

	test = isl_map_universe(isl_map_get_space(schedule_deps));
	test = isl_map_equate(test, isl_dim_out, dimension, isl_dim_in,
			      dimension);
	is_parallel = isl_map_is_subset(schedule_deps, test);

	isl_space_free(schedule_space);
	isl_map_free(test);
	isl_map_free(schedule_deps);

	return is_parallel;
}

/* Mark a for node openmp parallel, if it is the outermost parallel for node.
 */
static void mark_openmp_parallel(__isl_keep isl_ast_build *build,
	struct ast_build_userinfo *build_info,
	struct ast_node_userinfo *node_info)
{
	if (build_info->in_parallel_for)
		return;

	if (ast_schedule_dim_is_parallel(build, build_info->scop)) {
		build_info->in_parallel_for = 1;
		node_info->is_openmp = 1;
	}
}

/* Allocate an ast_node_info structure and initialize it with default values.
 */
static struct ast_node_userinfo *allocate_ast_node_userinfo()
{
	struct ast_node_userinfo *node_info;
	node_info = (struct ast_node_userinfo *)
		malloc(sizeof(struct ast_node_userinfo));
	node_info->is_openmp = 0;
	return node_info;
}

/* Free an ast_node_info structure.
 */
static void free_ast_node_userinfo(void *ptr)
{
	struct ast_node_userinfo *info;
	info = (struct ast_node_userinfo *) ptr;
	free(info);
}

/* This method is executed before the construction of a for node. It creates
 * an isl_id that is used to annotate the subsequently generated ast for nodes.
 *
 * In this function we also run the following analyses:
 *
 * 	- Detection of openmp parallel loops
 */
static __isl_give isl_id *ast_build_before_for(
	__isl_keep isl_ast_build *build, void *user)
{
	isl_id *id;
	struct ast_build_userinfo *build_info;
	struct ast_node_userinfo *node_info;

	build_info = (struct ast_build_userinfo *) user;
	node_info = allocate_ast_node_userinfo();
	id = isl_id_alloc(isl_ast_build_get_ctx(build), "", node_info);
	id = isl_id_set_free_user(id, free_ast_node_userinfo);

	mark_openmp_parallel(build, build_info, node_info);

	return id;
}

/* This method is executed after the construction of a for node.
 *
 * It performs the following actions:
 *
 * 	- Reset the 'in_parallel_for' flag, as soon as we leave a for node,
 * 	  that is marked as openmp parallel.
 *
 */
static __isl_give isl_ast_node *ast_build_after_for(
	__isl_take isl_ast_node *node, __isl_keep isl_ast_build *build,
	void *user)
{
	isl_id *id;
	struct ast_build_userinfo *build_info;
	struct ast_node_userinfo *info;

	id = isl_ast_node_get_annotation(node);
	info = isl_id_get_user(id);

	if (info && info->is_openmp) {
		build_info = (struct ast_build_userinfo *) user;
		build_info->in_parallel_for = 0;
	}

	isl_id_free(id);

	return node;
}

/* Find the element in scop->stmts that has the given "id".
 */
static struct pet_stmt *find_stmt(struct ppcg_scop *scop, __isl_keep isl_id *id)
{
	int i;

	for (i = 0; i < scop->pet->n_stmt; ++i) {
		struct pet_stmt *stmt = scop->pet->stmts[i];
		isl_id *id_i;

		id_i = isl_set_get_tuple_id(stmt->domain);
		isl_id_free(id_i);

		if (id_i == id)
			return stmt;
	}

	isl_die(isl_id_get_ctx(id), isl_error_internal,
		"statement not found", return NULL);
}

/* Print a user statement in the generated AST.
 * The ppcg_stmt has been attached to the node in at_each_domain.
 */
static __isl_give isl_printer *print_user(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	struct ppcg_stmt *stmt;
	isl_id *id;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	p = pet_stmt_print_body(stmt->stmt, p, stmt->ref2expr);

	isl_ast_print_options_free(print_options);

	return p;
}


/* Print a for loop node as an openmp parallel loop.
 *
 * To print an openmp parallel loop we print a normal for loop, but add
 * "#pragma openmp parallel for" in front.
 *
 * Variables that are declared within the body of this for loop are
 * automatically openmp 'private'. Iterators declared outside of the
 * for loop are automatically openmp 'shared'. As ppcg declares all iterators
 * at the position where they are assigned, there is no need to explicitly mark
 * variables. Their automatically assigned type is already correct.
 *
 * This function only generates valid OpenMP code, if the ast was generated
 * with the 'atomic-bounds' option enabled.
 *
 */
static __isl_give isl_printer *print_for_with_openmp(
	__isl_keep isl_ast_node *node, __isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "#pragma omp parallel for");
	p = isl_printer_end_line(p);

	p = isl_ast_node_for_print(node, p, print_options);

	return p;
}

/* Print a for node.
 *
 * Depending on how the node is annotated, we either print a normal
 * for node or an openmp parallel for node.
 */
static __isl_give isl_printer *print_for(__isl_take isl_printer *p,
	__isl_take isl_ast_print_options *print_options,
	__isl_keep isl_ast_node *node, void *user)
{
	isl_id *id;
	int openmp;

	openmp = 0;
	id = isl_ast_node_get_annotation(node);

	if (id) {
		struct ast_node_userinfo *info;

		info = (struct ast_node_userinfo *) isl_id_get_user(id);
		if (info && info->is_openmp)
			openmp = 1;
	}

	if (openmp)
		p = print_for_with_openmp(node, p, print_options);
	else
		p = isl_ast_node_for_print(node, p, print_options);

	isl_id_free(id);

	return p;
}

/* Index transformation callback for pet_stmt_build_ast_exprs.
 *
 * "index" expresses the array indices in terms of statement iterators
 * "iterator_map" expresses the statement iterators in terms of
 * AST loop iterators.
 *
 * The result expresses the array indices in terms of
 * AST loop iterators.
 */
static __isl_give isl_multi_pw_aff *pullback_index(
	__isl_take isl_multi_pw_aff *index, __isl_keep isl_id *id, void *user)
{
	isl_pw_multi_aff *iterator_map = user;

	iterator_map = isl_pw_multi_aff_copy(iterator_map);
	return isl_multi_pw_aff_pullback_pw_multi_aff(index, iterator_map);
}

/* Transform the accesses in the statement associated to the domain
 * called by "node" to refer to the AST loop iterators, construct
 * corresponding AST expressions using "build",
 * collect them in a ppcg_stmt and annotate the node with the ppcg_stmt.
 */
static __isl_give isl_ast_node *at_each_domain(__isl_take isl_ast_node *node,
	__isl_keep isl_ast_build *build, void *user)
{
	struct ppcg_scop *scop = user;
	isl_ast_expr *expr, *arg;
	isl_ctx *ctx;
	isl_id *id;
	isl_map *map;
	isl_pw_multi_aff *iterator_map;
	struct ppcg_stmt *stmt;

	ctx = isl_ast_node_get_ctx(node);
	stmt = isl_calloc_type(ctx, struct ppcg_stmt);
	if (!stmt)
		goto error;

	expr = isl_ast_node_user_get_expr(node);
	arg = isl_ast_expr_get_op_arg(expr, 0);
	isl_ast_expr_free(expr);
	id = isl_ast_expr_get_id(arg);
	isl_ast_expr_free(arg);
	stmt->stmt = find_stmt(scop, id);
	isl_id_free(id);
	if (!stmt->stmt)
		goto error;

	map = isl_map_from_union_map(isl_ast_build_get_schedule(build));
	map = isl_map_reverse(map);
	iterator_map = isl_pw_multi_aff_from_map(map);
	stmt->ref2expr = pet_stmt_build_ast_exprs(stmt->stmt, build,
				    &pullback_index, iterator_map, NULL, NULL);
	isl_pw_multi_aff_free(iterator_map);

	id = isl_id_alloc(isl_ast_node_get_ctx(node), NULL, stmt);
	id = isl_id_set_free_user(id, &ppcg_stmt_free);
	return isl_ast_node_set_annotation(node, id);
error:
	ppcg_stmt_free(stmt);
	return isl_ast_node_free(node);
}

/* Set *depth (initialized to 0 by the caller) to the maximum
 * of the schedule depths of the leaf nodes for which this function is called.
 */
static isl_bool update_depth(__isl_keep isl_schedule_node *node, void *user)
{
	int *depth = user;
	int node_depth;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_leaf)
		return isl_bool_true;
	node_depth = isl_schedule_node_get_schedule_depth(node);
	if (node_depth > *depth)
		*depth = node_depth;

	return isl_bool_false;
}

/* This function is called for each node in a CPU AST.
 * In case of a user node, print the macro definitions required
 * for printing the AST expressions in the annotation, if any.
 * For other nodes, return true such that descendants are also
 * visited.
 *
 * In particular, print the macro definitions needed for the substitutions
 * of the original user statements.
 */
static isl_bool at_node(__isl_keep isl_ast_node *node, void *user)
{
	struct ppcg_stmt *stmt;
	isl_id *id;
	isl_printer **p = user;

	if (isl_ast_node_get_type(node) != isl_ast_node_user)
		return isl_bool_true;

	id = isl_ast_node_get_annotation(node);
	stmt = isl_id_get_user(id);
	isl_id_free(id);

	if (!stmt)
		return isl_bool_error;

	*p = ppcg_print_body_macros(*p, stmt->ref2expr);
	if (!*p)
		return isl_bool_error;

	return isl_bool_false;
}

/* Print the required macros for the CPU AST "node" to "p",
 * including those needed for the user statements inside the AST.
 */
static __isl_give isl_printer *cpu_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	if (isl_ast_node_foreach_descendant_top_down(node, &at_node, &p) < 0)
		return isl_printer_free(p);
	p = ppcg_print_macros(p, node);
	return p;
}

/* Code generate the scop 'scop' using "schedule"
 * and print the corresponding C code to 'p'.
 */
static __isl_give isl_printer *print_scop(struct ppcg_scop *scop,
	__isl_take isl_schedule *schedule, __isl_take isl_printer *p,
	struct ppcg_options *options)
{
	isl_ctx *ctx = isl_printer_get_ctx(p);
	isl_ast_build *build;
	isl_ast_print_options *print_options;
	isl_ast_node *tree;
	isl_id_list *iterators;
	struct ast_build_userinfo build_info;
	int depth;

	depth = 0;
	if (isl_schedule_foreach_schedule_node_top_down(schedule, &update_depth,
						&depth) < 0)
		goto error;

	build = isl_ast_build_alloc(ctx);
	iterators = ppcg_scop_generate_names(scop, depth, "c");
	build = isl_ast_build_set_iterators(build, iterators);
	build = isl_ast_build_set_at_each_domain(build, &at_each_domain, scop);

	if (options->openmp) {
		build_info.scop = scop;
		build_info.in_parallel_for = 0;

		build = isl_ast_build_set_before_each_for(build,
							&ast_build_before_for,
							&build_info);
		build = isl_ast_build_set_after_each_for(build,
							&ast_build_after_for,
							&build_info);
	}

	tree = isl_ast_build_node_from_schedule(build, schedule);
	isl_ast_build_free(build);

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
							&print_user, NULL);

	print_options = isl_ast_print_options_set_print_for(print_options,
							&print_for, NULL);

	p = cpu_print_macros(p, tree);
	p = isl_ast_node_print(tree, p, print_options);

	isl_ast_node_free(tree);

	return p;
error:
	isl_schedule_free(schedule);
	isl_printer_free(p);
	return NULL;
}

/* Tile the band node "node" with tile sizes "sizes" and
 * mark all members of the resulting tile node as "atomic".
 */
static __isl_give isl_schedule_node *tile(__isl_take isl_schedule_node *node,
	__isl_take isl_multi_val *sizes)
{
	node = isl_schedule_node_band_tile(node, sizes);
	node = ppcg_set_schedule_node_type(node, isl_ast_loop_atomic);

	return node;
}

/* Tile "node", if it is a band node with at least 2 members.
 * The tile sizes are set from the "tile_size" option.
 */
static __isl_give isl_schedule_node *tile_band(
	__isl_take isl_schedule_node *node, void *user)
{
	struct ppcg_scop *scop = user;
	int n;
	isl_space *space;
	isl_multi_val *sizes;

	if (isl_schedule_node_get_type(node) != isl_schedule_node_band)
		return node;

	n = isl_schedule_node_band_n_member(node);
	if (n <= 1)
		return node;

	space = isl_schedule_node_band_get_space(node);
	sizes = ppcg_multi_val_from_int(space, scop->options->tile_size);

	return tile(node, sizes);
}

/* Construct schedule constraints from the dependences in ps
 * for the purpose of computing a schedule for a CPU.
 *
 * The proximity constraints are set to the flow dependences.
 *
 * If live-range reordering is allowed then the conditional validity
 * constraints are set to the order dependences with the flow dependences
 * as condition.  That is, a live-range (flow dependence) will be either
 * local to an iteration of a band or all adjacent order dependences
 * will be respected by the band.
 * The validity constraints are set to the union of the flow dependences
 * and the forced dependences, while the coincidence constraints
 * are set to the union of the flow dependences, the forced dependences and
 * the order dependences.
 *
 * If live-range reordering is not allowed, then both the validity
 * and the coincidence constraints are set to the union of the flow
 * dependences and the false dependences.
 *
 * Note that the coincidence constraints are only set when the "openmp"
 * options is set.  Even though the way openmp pragmas are introduced
 * does not rely on the coincident property of the schedule band members,
 * the coincidence constraints do affect the way the schedule is constructed,
 * such that more schedule dimensions should be detected as parallel
 * by ast_schedule_dim_is_parallel.
 * Since the order dependences are also taken into account by
 * ast_schedule_dim_is_parallel, they are also added to
 * the coincidence constraints.  If the openmp handling learns
 * how to privatize some memory, then the corresponding order
 * dependences can be removed from the coincidence constraints.
 */
static __isl_give isl_schedule_constraints *construct_cpu_schedule_constraints(
	struct ppcg_scop *ps)
{
	isl_schedule_constraints *sc;
	isl_union_map *validity, *coincidence;

	sc = isl_schedule_constraints_on_domain(isl_union_set_copy(ps->domain));
	if (ps->options->live_range_reordering) {
		sc = isl_schedule_constraints_set_conditional_validity(sc,
				isl_union_map_copy(ps->tagged_dep_flow),
				isl_union_map_copy(ps->tagged_dep_order));
		validity = isl_union_map_copy(ps->dep_flow);
		validity = isl_union_map_union(validity,
				isl_union_map_copy(ps->dep_forced));
		if (ps->options->openmp) {
			coincidence = isl_union_map_copy(validity);
			coincidence = isl_union_map_union(coincidence,
					isl_union_map_copy(ps->dep_order));
		}
	} else {
		validity = isl_union_map_copy(ps->dep_flow);
		validity = isl_union_map_union(validity,
				isl_union_map_copy(ps->dep_false));
		if (ps->options->openmp)
			coincidence = isl_union_map_copy(validity);
	}
	if (ps->options->openmp)
		sc = isl_schedule_constraints_set_coincidence(sc, coincidence);
	sc = isl_schedule_constraints_set_validity(sc, validity);
	sc = isl_schedule_constraints_set_proximity(sc,
					isl_union_map_copy(ps->dep_flow));

	return sc;
}

/* Compute a schedule for the scop "ps".
 *
 * First derive the appropriate schedule constraints from the dependences
 * in "ps" and then compute a schedule from those schedule constraints,
 * possibly grouping statement instances based on the input schedule.
 */
static __isl_give isl_schedule *compute_cpu_schedule(struct ppcg_scop *ps)
{
	isl_schedule_constraints *sc;
	isl_schedule *schedule;

	if (!ps)
		return NULL;

	sc = construct_cpu_schedule_constraints(ps);

	if (ps->options->debug->dump_schedule_constraints)
		isl_schedule_constraints_dump(sc);
	schedule = ppcg_compute_schedule(sc, ps->schedule, ps->options);

	return schedule;
}

/* Compute a new schedule to the scop "ps" if the reschedule option is set.
 * Otherwise, return a copy of the original schedule.
 */
static __isl_give isl_schedule *optionally_compute_schedule(void *user)
{
	struct ppcg_scop *ps = user;

	if (!ps)
		return NULL;
	if (!ps->options->reschedule)
		return isl_schedule_copy(ps->schedule);
	return compute_cpu_schedule(ps);
}

/* Compute a schedule based on the dependences in "ps" and
 * tile it if requested by the user.
 */
static __isl_give isl_schedule *get_schedule(struct ppcg_scop *ps,
	struct ppcg_options *options)
{
	isl_ctx *ctx;
	isl_schedule *schedule;

	if (!ps)
		return NULL;

	ctx = isl_union_set_get_ctx(ps->domain);
	schedule = ppcg_get_schedule(ctx, options,
				    &optionally_compute_schedule, ps);
	if (ps->options->tile)
		schedule = isl_schedule_map_schedule_node_bottom_up(schedule,
							&tile_band, ps);

	return schedule;
}

/* Generate CPU code for the scop "ps" using "schedule" and
 * print the corresponding C code to "p", including variable declarations.
 */
static __isl_give isl_printer *print_cpu_with_schedule(
	__isl_take isl_printer *p, struct ppcg_scop *ps,
	__isl_take isl_schedule *schedule, struct ppcg_options *options)
{
	int hidden;
	isl_set *context;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "/* ppcg generated CPU code */");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	p = ppcg_set_macro_names(p);
	p = ppcg_print_exposed_declarations(p, ps);
	hidden = ppcg_scop_any_hidden_declarations(ps);
	if (hidden) {
		p = ppcg_start_block(p);
		p = ppcg_print_hidden_declarations(p, ps);
	}

	context = isl_set_copy(ps->context);
	context = isl_set_from_params(context);
	schedule = isl_schedule_insert_context(schedule, context);
	if (options->debug->dump_final_schedule)
		isl_schedule_dump(schedule);
	p = print_scop(ps, schedule, p, options);
	if (hidden)
		p = ppcg_end_block(p);

	return p;
}

/* Generate CPU code for the scop "ps" and print the corresponding C code
 * to "p", including variable declarations.
 */
__isl_give isl_printer *print_cpu(__isl_take isl_printer *p,
	struct ppcg_scop *ps, struct ppcg_options *options)
{
	isl_schedule *schedule;

	schedule = isl_schedule_copy(ps->schedule);
	return print_cpu_with_schedule(p, ps, schedule, options);
}

/* Generate CPU code for "scop" and print it to "p".
 *
 * First obtain a schedule for "scop" and then print code for "scop"
 * using that schedule.
 */
static __isl_give isl_printer *generate(__isl_take isl_printer *p,
	struct ppcg_scop *scop, struct ppcg_options *options)
{
	isl_schedule *schedule;

	schedule = get_schedule(scop, options);

	return print_cpu_with_schedule(p, scop, schedule, options);
}

/* Wrapper around generate for use as a ppcg_transform callback.
 */
static __isl_give isl_printer *print_cpu_wrap(__isl_take isl_printer *p,
	struct ppcg_scop *scop, void *user)
{
	struct ppcg_options *options = user;

	return generate(p, scop, options);
}

/* Transform the code in the file called "input" by replacing
 * all scops by corresponding CPU code and write the results to a file
 * called "output".
 */
int generate_cpu(isl_ctx *ctx, struct ppcg_options *options,
	const char *input, const char *output)
{
	FILE *output_file;
	int r;

	output_file = get_output_file(input, output);
	if (!output_file)
		return -1;

	r = ppcg_transform(ctx, input, output_file, options,
					&print_cpu_wrap, options);

	fclose(output_file);

	return r;
}
