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
	isl_ast_build *build;

	if (!prog)
		return isl_printer_free(p);

	build = isl_ast_build_from_context(isl_set_copy(prog->scop->context));
	for (i = 0; i < prog->n_array; ++i) {
		if (!prog->array[i].declare_local)
			continue;
		p = ppcg_print_declaration(p, prog->scop->pet->arrays[i],
					    build);
	}
	isl_ast_build_free(build);

	return p;
}

/* Print an expression for the size of "array" in bytes.
 */
__isl_give isl_printer *gpu_array_info_print_size(__isl_take isl_printer *prn,
	struct gpu_array_info *array)
{
	int i;

	for (i = 0; i < array->n_index; ++i) {
		prn = isl_printer_print_str(prn, "(");
		prn = isl_printer_print_pw_aff(prn, array->bound[i]);
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
	int i;

	p = isl_printer_print_str(p, array->type);
	p = isl_printer_print_str(p, " ");

	p = isl_printer_print_str(p, array->name);

	for (i = 0; i < array->n_index; i++) {
		p = isl_printer_print_str(p, "[");
		p = isl_printer_print_pw_aff(p, array->bound[i]);
		p = isl_printer_print_str(p, "]");
	}

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
 *
 * The copy in global memory has been linearized, so we need to take
 * the array size into account.
 */
static __isl_give isl_printer *stmt_print_global_index(
	__isl_take isl_printer *p, struct ppcg_kernel_stmt *stmt)
{
	int i;
	struct gpu_array_info *array = stmt->u.c.array;
	struct gpu_local_array_info *local = stmt->u.c.local_array;
	isl_ast_expr *index;

	if (gpu_array_is_scalar(array)) {
		if (!gpu_array_is_read_only_scalar(array))
			p = isl_printer_print_str(p, "*");
		p = isl_printer_print_str(p, array->name);
		return p;
	}

	index = isl_ast_expr_copy(stmt->u.c.index);
	if (array->linearize)
		index = gpu_local_array_info_linearize_index(local, index);

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
