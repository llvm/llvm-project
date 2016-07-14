/*
 * Copyright 2012 INRIA Paris-Rocquencourt
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Tobias Grosser, INRIA Paris-Rocquencourt,
 * Domaine de Voluceau, Rocquenqourt, B.P. 105,
 * 78153 Le Chesnay Cedex France
 */

#include <limits.h>
#include <stdio.h>
#include <string.h>

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/ast_build.h>
#include <pet.h>

#include "ppcg.h"
#include "ppcg_options.h"
#include "cpu.h"
#include "print.h"

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
	int i;

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
	isl_union_map *schedule_node, *schedule, *deps;
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
static __isl_give isl_ast_node *ast_build_after_for(__isl_take isl_ast_node *node,
        __isl_keep isl_ast_build *build, void *user) {
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
	struct ppcg_print_info *print_info;
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

/* Set *depth to the number of scheduling dimensions
 * for the schedule of the first domain.
 * We assume here that this number is the same for all domains.
 */
static isl_stat set_depth(__isl_take isl_map *map, void *user)
{
	unsigned *depth = user;

	*depth = isl_map_dim(map, isl_dim_out);

	isl_map_free(map);
	return isl_stat_error;
}

/* Code generate the scop 'scop' and print the corresponding C code to 'p'.
 */
static __isl_give isl_printer *print_scop(struct ppcg_scop *scop,
	__isl_take isl_printer *p, struct ppcg_options *options)
{
	isl_ctx *ctx = isl_printer_get_ctx(p);
	isl_set *context;
	isl_union_set *domain_set;
	isl_union_map *schedule_map;
	isl_ast_build *build;
	isl_ast_print_options *print_options;
	isl_ast_node *tree;
	isl_id_list *iterators;
	struct ast_build_userinfo build_info;
	int depth;

	context = isl_set_copy(scop->context);
	domain_set = isl_union_set_copy(scop->domain);
	schedule_map = isl_schedule_get_map(scop->schedule);
	schedule_map = isl_union_map_intersect_domain(schedule_map, domain_set);

	isl_union_map_foreach_map(schedule_map, &set_depth, &depth);

	build = isl_ast_build_from_context(context);
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

	tree = isl_ast_build_node_from_schedule_map(build, schedule_map);
	isl_ast_build_free(build);

	print_options = isl_ast_print_options_alloc(ctx);
	print_options = isl_ast_print_options_set_print_user(print_options,
							&print_user, NULL);

	print_options = isl_ast_print_options_set_print_for(print_options,
							&print_for, NULL);

	p = ppcg_print_macros(p, tree);
	p = isl_ast_node_print(tree, p, print_options);

	isl_ast_node_free(tree);

	return p;
}

/* Generate CPU code for the scop "ps" and print the corresponding C code
 * to "p", including variable declarations.
 */
__isl_give isl_printer *print_cpu(__isl_take isl_printer *p,
	struct ppcg_scop *ps, struct ppcg_options *options)
{
	int hidden;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "/* ppcg generated CPU code */");
	p = isl_printer_end_line(p);

	p = isl_printer_start_line(p);
	p = isl_printer_end_line(p);

	p = isl_ast_op_type_print_macro(isl_ast_op_fdiv_q, p);
	p = ppcg_print_exposed_declarations(p, ps);
	hidden = ppcg_scop_any_hidden_declarations(ps);
	if (hidden) {
		p = ppcg_start_block(p);
		p = ppcg_print_hidden_declarations(p, ps);
	}
	if (options->debug->dump_final_schedule)
		isl_schedule_dump(ps->schedule);
	p = print_scop(ps, p, options);
	if (hidden)
		p = ppcg_end_block(p);

	return p;
}

/* Wrapper around print_cpu for use as a ppcg_transform callback.
 */
static __isl_give isl_printer *print_cpu_wrap(__isl_take isl_printer *p,
	struct ppcg_scop *scop, void *user)
{
	struct ppcg_options *options = user;

	return print_cpu(p, scop, options);
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
