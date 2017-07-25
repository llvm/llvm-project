/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <string.h>

#include <isl/aff.h>

#include "gpu_print.h"
#include "print.h"
#include "schedule.h"

/* Print declarations to "p" for arrays that are local to "prog"
 * but that are used on the host and therefore require a declaration.
 */
__isl_give isl_printer *gpu_print_local_declarations(__isl_take isl_printer *p,
	struct gpu_prog *prog)
{
	int i;

	if (!prog)
		return isl_printer_free(p);

	for (i = 0; i < prog->n_array; ++i) {
		struct gpu_array_info *array = &prog->array[i];
		isl_ast_expr *size;

		if (!array->declare_local)
			continue;
		size = array->declared_size;
		p = ppcg_print_declaration_with_size(p, array->type, size);
	}

	return p;
}

/* Print an expression for the size of "array" in bytes.
 */
__isl_give isl_printer *gpu_array_info_print_size(__isl_take isl_printer *prn,
	struct gpu_array_info *array)
{
	int i;

	for (i = 0; i < array->n_index; ++i) {
		isl_ast_expr *bound;

		prn = isl_printer_print_str(prn, "(");
		bound = isl_ast_expr_get_op_arg(array->bound_expr, 1 + i);
		prn = isl_printer_print_ast_expr(prn, bound);
		isl_ast_expr_free(bound);
		prn = isl_printer_print_str(prn, ") * ");
	}
	prn = isl_printer_print_str(prn, "sizeof(");
	prn = isl_printer_print_str(prn, array->type);
	prn = isl_printer_print_str(prn, ")");

	return prn;
}

/* Print the declaration of a non-linearized array argument.
 */
static __isl_give isl_printer *print_non_linearized_declaration_argument(
	__isl_take isl_printer *p, struct gpu_array_info *array)
{
	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");

	p = isl_printer_print_ast_expr(p, array->bound_expr);

	return p;
}

/* Print the declaration of an array argument.
 * "memory_space" allows to specify a memory space prefix.
 */
__isl_give isl_printer *gpu_array_info_print_declaration_argument(
	__isl_take isl_printer *p, struct gpu_array_info *array,
	const char *memory_space)
{
	if (gpu_array_is_read_only_scalar(array)) {
		p = isl_printer_print_str(p, array->type);
		p = isl_printer_print_str(p, " ");
		p = isl_printer_print_str(p, array->name);
		return p;
	}

	if (memory_space) {
		p = isl_printer_print_str(p, memory_space);
		p = isl_printer_print_str(p, " ");
	}

	if (array->n_index != 0 && !array->linearize)
		return print_non_linearized_declaration_argument(p, array);

	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_str(p, "*");
	p = isl_printer_print_str(p, array->name);

	return p;
}

/* Print the call of an array argument.
 */
__isl_give isl_printer *gpu_array_info_print_call_argument(
	__isl_take isl_printer *p, struct gpu_array_info *array)
{
	if (gpu_array_is_read_only_scalar(array))
		return isl_printer_print_str(p, array->name);

	p = isl_printer_print_str(p, "dev_");
	p = isl_printer_print_str(p, array->name);

	return p;
}

/* Print an access to the element in the private/shared memory copy
 * described by "stmt".  The index of the copy is recorded in
 * stmt->local_index as an access to the array.
 */
static __isl_give isl_printer *stmt_print_local_index(__isl_take isl_printer *p,
	struct ppcg_kernel_stmt *stmt)
{
	return isl_printer_print_ast_expr(p, stmt->u.c.local_index);
}

/* Print an access to the element in the global memory copy
 * described by "stmt".  The index of the copy is recorded in
 * stmt->index as an access to the array.
 */
static __isl_give isl_printer *stmt_print_global_index(
	__isl_take isl_printer *p, struct ppcg_kernel_stmt *stmt)
{
	struct gpu_array_info *array = stmt->u.c.array;
	isl_ast_expr *index;

	if (gpu_array_is_scalar(array)) {
		if (!gpu_array_is_read_only_scalar(array))
			p = isl_printer_print_str(p, "*");
		p = isl_printer_print_str(p, array->name);
		return p;
	}

	index = isl_ast_expr_copy(stmt->u.c.index);

	p = isl_printer_print_ast_expr(p, index);
	isl_ast_expr_free(index);

	return p;
}

/* Print a copy statement.
 *
 * A read copy statement is printed as
 *
 *	local = global;
 *
 * while a write copy statement is printed as
 *
 *	global = local;
 */
__isl_give isl_printer *ppcg_kernel_print_copy(__isl_take isl_printer *p,
	struct ppcg_kernel_stmt *stmt)
{
	p = isl_printer_start_line(p);
	if (stmt->u.c.read) {
		p = stmt_print_local_index(p, stmt);
		p = isl_printer_print_str(p, " = ");
		p = stmt_print_global_index(p, stmt);
	} else {
		p = stmt_print_global_index(p, stmt);
		p = isl_printer_print_str(p, " = ");
		p = stmt_print_local_index(p, stmt);
	}
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

__isl_give isl_printer *ppcg_kernel_print_domain(__isl_take isl_printer *p,
	struct ppcg_kernel_stmt *stmt)
{
	return pet_stmt_print_body(stmt->u.d.stmt->stmt, p, stmt->u.d.ref2expr);
}

/* This function is called for each node in a GPU AST.
 * In case of a user node, print the macro definitions required
 * for printing the AST expressions in the annotation, if any.
 * For other nodes, return true such that descendants are also
 * visited.
 *
 * In particular, for a kernel launch, print the macro definitions
 * needed for the grid size.
 * For a copy statement, print the macro definitions needed
 * for the two index expressions.
 * For an original user statement, print the macro definitions
 * needed for the substitutions.
 */
static isl_bool at_node(__isl_keep isl_ast_node *node, void *user)
{
	const char *name;
	isl_id *id;
	int is_kernel;
	struct ppcg_kernel *kernel;
	struct ppcg_kernel_stmt *stmt;
	isl_printer **p = user;

	if (isl_ast_node_get_type(node) != isl_ast_node_user)
		return isl_bool_true;

	id = isl_ast_node_get_annotation(node);
	if (!id)
		return isl_bool_false;

	name = isl_id_get_name(id);
	if (!name)
		return isl_bool_error;
	is_kernel = !strcmp(name, "kernel");
	kernel = is_kernel ? isl_id_get_user(id) : NULL;
	stmt = is_kernel ? NULL : isl_id_get_user(id);
	isl_id_free(id);

	if ((is_kernel && !kernel) || (!is_kernel && !stmt))
		return isl_bool_error;

	if (is_kernel) {
		*p = ppcg_ast_expr_print_macros(kernel->grid_size_expr, *p);
	} else if (stmt->type == ppcg_kernel_copy) {
		*p = ppcg_ast_expr_print_macros(stmt->u.c.index, *p);
		*p = ppcg_ast_expr_print_macros(stmt->u.c.local_index, *p);
	} else if (stmt->type == ppcg_kernel_domain) {
		*p = ppcg_print_body_macros(*p, stmt->u.d.ref2expr);
	}
	if (!*p)
		return isl_bool_error;

	return isl_bool_false;
}

/* Print the required macros for the GPU AST "node" to "p",
 * including those needed for the user statements inside the AST.
 */
__isl_give isl_printer *gpu_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	if (isl_ast_node_foreach_descendant_top_down(node, &at_node, &p) < 0)
		return isl_printer_free(p);
	p = ppcg_print_macros(p, node);
	return p;
}

/* Was the definition of "type" printed before?
 * That is, does its name appear in the list of printed types "types"?
 */
static int already_printed(struct gpu_types *types,
	struct pet_type *type)
{
	int i;

	for (i = 0; i < types->n; ++i)
		if (!strcmp(types->name[i], type->name))
			return 1;

	return 0;
}

/* Print the definitions of all types prog->scop that have not been
 * printed before (according to "types") on "p".
 * Extend the list of printed types "types" with the newly printed types.
 */
__isl_give isl_printer *gpu_print_types(__isl_take isl_printer *p,
	struct gpu_types *types, struct gpu_prog *prog)
{
	int i, n;
	isl_ctx *ctx;
	char **name;

	n = prog->scop->pet->n_type;

	if (n == 0)
		return p;

	ctx = isl_printer_get_ctx(p);
	name = isl_realloc_array(ctx, types->name, char *, types->n + n);
	if (!name)
		return isl_printer_free(p);
	types->name = name;

	for (i = 0; i < n; ++i) {
		struct pet_type *type = prog->scop->pet->types[i];

		if (already_printed(types, type))
			continue;

		p = isl_printer_start_line(p);
		p = isl_printer_print_str(p, type->definition);
		p = isl_printer_print_str(p, ";");
		p = isl_printer_end_line(p);

		types->name[types->n++] = strdup(type->name);
	}

	return p;
}
