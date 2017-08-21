#ifndef PRINT_H
#define PRINT_H

#include <isl/ast.h>

#include "ppcg.h"

__isl_give isl_printer *ppcg_start_block(__isl_take isl_printer *p);
__isl_give isl_printer *ppcg_end_block(__isl_take isl_printer *p);

__isl_give isl_printer *ppcg_print_macros(__isl_take isl_printer *p,
	__isl_keep isl_ast_node *node);

__isl_give isl_printer *ppcg_print_declaration(__isl_take isl_printer *p,
	struct pet_array *array, __isl_keep isl_ast_build *build);
__isl_give isl_printer *ppcg_print_exposed_declarations(
	__isl_take isl_printer *p, struct ppcg_scop *scop);
__isl_give isl_printer *ppcg_print_hidden_declarations(
	__isl_take isl_printer *p, struct ppcg_scop *scop);

__isl_give isl_printer *ppcg_print_guarded(__isl_take isl_printer *p,
	__isl_take isl_set *guard, __isl_take isl_set *context,
	__isl_give isl_printer *(*fn)(__isl_take isl_printer *p, void *user),
	void *user);

#endif
