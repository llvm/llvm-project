/*
 * Copyright 2012-2013 Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d’Ulm, 75230 Paris, France
 */

#include <isl/aff.h>
#include <isl/ast_build.h>
#include <isl/id.h>

#include "print.h"
#include "util.h"

__isl_give isl_printer *ppcg_start_block(__isl_take isl_printer *p)
{
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "{");
	p = isl_printer_end_line(p);
	p = isl_printer_indent(p, 2);
	return p;
}

__isl_give isl_printer *ppcg_end_block(__isl_take isl_printer *p)
{
	p = isl_printer_indent(p, -2);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "}");
	p = isl_printer_end_line(p);
	return p;
}

/* Names of notes that keep track of whether min/max
 * macro definitions have already been printed.
 */
static const char *ppcg_max_printed = "ppcg_max_printed";
static const char *ppcg_min_printed = "ppcg_min_printed";

/* Has the macro definition corresponding to "note_name" been printed
 * to "p" before?
 * That is, does "p" have an associated "note_name" note?
 */
static isl_bool printed_before(__isl_keep isl_printer *p, const char *note_name)
{
	isl_ctx *ctx;
	isl_id *id;
	isl_bool printed;

	if (!p)
		return isl_bool_error;

	ctx = isl_printer_get_ctx(p);
	id = isl_id_alloc(ctx, note_name, NULL);
	printed = isl_printer_has_note(p, id);
	isl_id_free(id);

	return printed;
}

/* Keep track of the fact that the macro definition corresponding
 * to "note_name" has been printed to "p" by attaching a note with
 * that name.  The value of the note is of no importance, but it
 * has to be a valid isl_id, so the note identifier is reused
 * as the note.
 */
static __isl_give isl_printer *mark_printed(__isl_take isl_printer *p,
	const char *note_name)
{
	isl_ctx *ctx;
	isl_id *id;

	if (!p)
		return NULL;

	ctx = isl_printer_get_ctx(p);
	id = isl_id_alloc(ctx, note_name, NULL);
	return isl_printer_set_note(p, id, isl_id_copy(id));
}

/* Print a macro definition "def" for the macro "name" to "p",
 * unless such a macro definition has been printed to "p" before.
 * "note_name" is used as the name of the note that keeps track
 * of whether this printing has happened.
 */
static __isl_give isl_printer *print_ppcg_macro(__isl_take isl_printer *p,
	const char *name, const char *def, const char *note_name)
{
	isl_bool printed;

	printed = printed_before(p, note_name);
	if (printed < 0)
		return isl_printer_free(p);
	if (printed)
		return p;

	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, "#define ");
	p = isl_printer_print_str(p, name);
	p = isl_printer_print_str(p, def);
	p = isl_printer_end_line(p);

	p = mark_printed(p, note_name);

	return p;
}

/* Structure for keeping track of definitions of some macros.
 */
struct ppcg_macros {
	const char *min;
	const char *max;
};

/* Free the memory allocated by a struct ppcg_macros.
 */
static void ppcg_macros_free(void *user)
{
	free(user);
}

/* Default macro definitions (when GNU extensions are allowed).
 */
struct ppcg_macros ppcg_macros_default = {
	.min = "(x,y)    "
		"({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); "
		"_x < _y ? _x : _y; })",
	.max = "(x,y)    "
		"({ __typeof__(x) _x = (x); __typeof__(y) _y = (y); "
		"_x > _y ? _x : _y; })",
};

/* Name used for the note that keeps track of macro definitions.
 */
static const char *ppcg_macros = "ppcg_macros";

/* Set the macro definitions for isl_ast_op_min and isl_ast_op_max
 * to "min" and "max" and store them in "p".
 *
 * In particular, create a ppcg_macros object and attach it
 * as a note to the printer.
 */
__isl_give isl_printer *ppcg_set_macros(__isl_take isl_printer *p,
	const char *min, const char *max)
{
	isl_ctx *ctx;
	isl_id *id, *macros_id;
	struct ppcg_macros *macros;

	if (!p)
		return NULL;

	ctx = isl_printer_get_ctx(p);
	macros = isl_alloc_type(ctx, struct ppcg_macros);
	if (!macros)
		return isl_printer_free(p);
	macros->min = min;
	macros->max = max;
	id = isl_id_alloc(ctx, ppcg_macros, NULL);
	macros_id = isl_id_alloc(ctx, NULL, macros);
	if (!macros_id)
		ppcg_macros_free(macros);
	else
		macros_id = isl_id_set_free_user(macros_id, &ppcg_macros_free);

	p = isl_printer_set_note(p, id, macros_id);

	return p;
}

/* Return the ppcg_macros object that holds the currently active
 * macro definitions in "p".
 * If "p" has a note with macro definitions, then return those.
 * Otherwise, return the default macro definitions.
 */
static struct ppcg_macros *get_macros(__isl_keep isl_printer *p)
{
	isl_id *id;
	isl_bool has_macros;
	struct ppcg_macros *macros;

	id = isl_id_alloc(isl_printer_get_ctx(p), ppcg_macros, NULL);
	has_macros = isl_printer_has_note(p, id);
	if (has_macros < 0 || !has_macros) {
		isl_id_free(id);
		if (has_macros < 0)
			return NULL;
		return &ppcg_macros_default;
	}
	id = isl_printer_get_note(p, id);
	macros = isl_id_get_user(id);
	isl_id_free(id);

	return macros;
}

/* Print the currently active macro definition for ppcg_max.
 */
static __isl_give isl_printer *print_max(__isl_take isl_printer *p)
{
	struct ppcg_macros *macros;

	macros = get_macros(p);
	if (!macros)
		return isl_printer_free(p);
	return print_ppcg_macro(p, ppcg_max, macros->max, ppcg_max_printed);
}

/* Print the currently active macro definition for ppcg_min.
 */
static __isl_give isl_printer *print_min(__isl_take isl_printer *p)
{
	struct ppcg_macros *macros;

	macros = get_macros(p);
	if (!macros)
		return isl_printer_free(p);
	return print_ppcg_macro(p, ppcg_min, macros->min, ppcg_min_printed);
}

/* Print a macro definition for "type" to "p".
 * If GNU extensions are allowed, then print a specialized definition
 * for isl_ast_op_min and isl_ast_op_max.
 * Otherwise, use the default isl definition.
 */
__isl_give isl_printer *ppcg_print_macro(enum isl_ast_op_type type,
	__isl_take isl_printer *p)
{
	isl_ctx *ctx;
	struct ppcg_options *options;

	if (!p)
		return NULL;

	ctx = isl_printer_get_ctx(p);
	options = isl_ctx_peek_options(ctx, &ppcg_options_args);
	if (!options || !options->allow_gnu_extensions)
		return isl_ast_op_type_print_macro(type, p);

	switch (type) {
	case isl_ast_op_max:
		return print_max(p);
	case isl_ast_op_min:
		return print_min(p);
	default:
		return isl_ast_op_type_print_macro(type, p);
	}
}

/* isl_ast_expr_foreach_ast_op_type or isl_ast_node_foreach_ast_op_type
 * callback that prints a macro definition for "type".
 */
static isl_stat print_macro(enum isl_ast_op_type type, void *user)
{
	isl_printer **p = user;

	*p = ppcg_print_macro(type, *p);
	if (!*p)
		return isl_stat_error;

	return isl_stat_ok;
}

/* Print the required macros for "expr".
 */
__isl_give isl_printer *ppcg_ast_expr_print_macros(
	__isl_keep isl_ast_expr *expr, __isl_take isl_printer *p)
{
	if (isl_ast_expr_foreach_ast_op_type(expr, &print_macro, &p) < 0)
		return isl_printer_free(p);
	return p;
}

/* isl_id_to_ast_expr_foreach callback that prints the required
 * macro definitions for "val".
 */
static isl_stat print_expr_macros(__isl_take isl_id *key,
	__isl_take isl_ast_expr *val, void *user)
{
	isl_printer **p = user;

	*p = ppcg_ast_expr_print_macros(val, *p);
	isl_id_free(key);
	isl_ast_expr_free(val);

	if (!*p)
		return isl_stat_error;
	return isl_stat_ok;
}

/* Print the required macro definitions for the body of a statement in which
 * the access expressions are replaced by the isl_ast_expr objects
 * in "ref2expr".
 */
__isl_give isl_printer *ppcg_print_body_macros(__isl_take isl_printer *p,
	__isl_keep isl_id_to_ast_expr *ref2expr)
{
	if (isl_id_to_ast_expr_foreach(ref2expr, &print_expr_macros, &p) < 0)
		return isl_printer_free(p);
	return p;
}

/* Print the required macros for "node".
 */
__isl_give isl_printer *ppcg_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node)
{
	if (isl_ast_node_foreach_ast_op_type(node, &print_macro, &p) < 0)
		return isl_printer_free(p);
	return p;
}

/* Names used for the macros that may appear in a printed isl AST.
 */
const char *ppcg_min = "ppcg_min";
const char *ppcg_max = "ppcg_max";
const char *ppcg_fdiv_q = "ppcg_fdiv_q";

/* Set the names of the macros that may appear in a printed isl AST.
 */
__isl_give isl_printer *ppcg_set_macro_names(__isl_take isl_printer *p)
{
	p = isl_ast_op_type_set_print_name(p, isl_ast_op_min, ppcg_min);
	p = isl_ast_op_type_set_print_name(p, isl_ast_op_max, ppcg_max);
	p = isl_ast_op_type_set_print_name(p, isl_ast_op_fdiv_q, ppcg_fdiv_q);

	return p;
}

/* Given a multi affine expression "mpa" without domain, modify it to have
 * the schedule space of "build" as domain.
 *
 * If the schedule space of "build" is a parameter space, then nothing
 * needs to be done.
 * Otherwise, "mpa" is first given a 0D domain and then it is combined
 * with a mapping from the schedule space of "build" to the same 0D domain.
 */
__isl_give isl_multi_pw_aff *ppcg_attach_multi_pw_aff(
	__isl_take isl_multi_pw_aff *mpa, __isl_keep isl_ast_build *build)
{
	isl_bool params;
	isl_space *space;
	isl_multi_aff *ma;

	space = isl_ast_build_get_schedule_space(build);
	params = isl_space_is_params(space);
	if (params < 0 || params) {
		isl_space_free(space);
		if (params < 0)
			return isl_multi_pw_aff_free(mpa);
		return mpa;
	}
	space = isl_space_from_domain(space);
	ma = isl_multi_aff_zero(space);
	mpa = isl_multi_pw_aff_from_range(mpa);
	mpa = isl_multi_pw_aff_pullback_multi_aff(mpa, ma);

	return mpa;
}

/* Build an access AST expression from "size" using "build".
 * "size" does not have a domain, but "build" may have a proper schedule space.
 * First modify "size" to have that schedule space as domain.
 */
__isl_give isl_ast_expr *ppcg_build_size_expr(__isl_take isl_multi_pw_aff *size,
	__isl_keep isl_ast_build *build)
{
	size = ppcg_attach_multi_pw_aff(size, build);
	return isl_ast_build_access_from_multi_pw_aff(build, size);
}

/* Print a declaration for an array with element type "base_type" and
 * size "size" to "p".
 */
__isl_give isl_printer *ppcg_print_declaration_with_size(
	__isl_take isl_printer *p, const char *base_type,
	__isl_keep isl_ast_expr *size)
{
	if (!base_type || !size)
		return isl_printer_free(p);

	p = ppcg_ast_expr_print_macros(size, p);
	p = isl_printer_start_line(p);
	p = isl_printer_print_str(p, base_type);
	p = isl_printer_print_str(p, " ");
	p = isl_printer_print_ast_expr(p, size);
	p = isl_printer_print_str(p, ";");
	p = isl_printer_end_line(p);

	return p;
}

/* Print a declaration for array "array" to "p", using "build"
 * to simplify any size expressions.
 *
 * The size is computed from the extent of the array and is
 * subsequently converted to an "access expression" by "build".
 */
__isl_give isl_printer *ppcg_print_declaration(__isl_take isl_printer *p,
	struct pet_array *array, __isl_keep isl_ast_build *build)
{
	isl_multi_pw_aff *size;
	isl_ast_expr *expr;

	if (!array)
		return isl_printer_free(p);

	size = ppcg_size_from_extent(isl_set_copy(array->extent));
	expr = isl_ast_build_access_from_multi_pw_aff(build, size);
	p = ppcg_print_declaration_with_size(p, array->element_type, expr);
	isl_ast_expr_free(expr);

	return p;
}

/* Print declarations for the arrays in "scop" that are declared
 * and that are exposed (if exposed == 1) or not exposed (if exposed == 0).
 */
static __isl_give isl_printer *print_declarations(__isl_take isl_printer *p,
	struct ppcg_scop *scop, int exposed)
{
	int i;
	isl_ast_build *build;

	if (!scop)
		return isl_printer_free(p);

	build = isl_ast_build_from_context(isl_set_copy(scop->context));
	for (i = 0; i < scop->pet->n_array; ++i) {
		struct pet_array *array = scop->pet->arrays[i];

		if (!array->declared)
			continue;
		if (array->exposed != exposed)
			continue;

		p = ppcg_print_declaration(p, array, build);
	}
	isl_ast_build_free(build);

	return p;
}

/* Print declarations for the arrays in "scop" that are declared
 * and exposed to the code after the scop.
 */
__isl_give isl_printer *ppcg_print_exposed_declarations(
	__isl_take isl_printer *p, struct ppcg_scop *scop)
{
	return print_declarations(p, scop, 1);
}

/* Print declarations for the arrays in "scop" that are declared,
 * but not exposed to the code after the scop.
 */
__isl_give isl_printer *ppcg_print_hidden_declarations(
	__isl_take isl_printer *p, struct ppcg_scop *scop)
{
	return print_declarations(p, scop, 0);
}
