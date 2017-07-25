#ifndef PRINT_H
#define PRINT_H

#include <isl/ast.h>

#include "ppcg.h"

extern const char *ppcg_min;
extern const char *ppcg_max;
extern const char *ppcg_fdiv_q;

__isl_give isl_printer *ppcg_start_block(__isl_take isl_printer *p);
__isl_give isl_printer *ppcg_end_block(__isl_take isl_printer *p);

__isl_give isl_printer *ppcg_set_macro_names(__isl_take isl_printer *p);
__isl_give isl_printer *ppcg_set_macros(__isl_take isl_printer *p,
	const char *min, const char *max);
__isl_give isl_printer *ppcg_print_macro(enum isl_ast_op_type type,
	__isl_take isl_printer *p);
__isl_give isl_printer *ppcg_ast_expr_print_macros(
	__isl_keep isl_ast_expr *expr, __isl_take isl_printer *p);
__isl_give isl_printer *ppcg_print_body_macros(__isl_take isl_printer *p,
	__isl_keep isl_id_to_ast_expr *ref2expr);
__isl_give isl_printer *ppcg_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node);

__isl_give isl_ast_expr *ppcg_build_size_expr(__isl_take isl_multi_pw_aff *size,
	__isl_keep isl_ast_build *build);

__isl_give isl_printer *ppcg_print_declaration_with_size(
	__isl_take isl_printer *p, const char *base_type,
	__isl_keep isl_ast_expr *size);
__isl_give isl_printer *ppcg_print_declaration(__isl_take isl_printer *p,
	struct pet_array *array, __isl_keep isl_ast_build *build);
__isl_give isl_printer *ppcg_print_exposed_declarations(
	__isl_take isl_printer *p, struct ppcg_scop *scop);
__isl_give isl_printer *ppcg_print_hidden_declarations(
	__isl_take isl_printer *p, struct ppcg_scop *scop);

#endif
