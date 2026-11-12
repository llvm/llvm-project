/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Replace the field FIELD of "node" by "field",
 * where the field may or may not have already been set in "node".
 * However, if the field has not already been set,
 * then "node" is required to have a single reference.
 * In this case the call to isl_ast_node_cow has no effect.
 */
__isl_give isl_ast_node *FN(FN(FN(isl_ast_node,NODE_TYPE),set),FIELD_NAME)(
	__isl_take isl_ast_node *node, __isl_take FIELD_TYPE *field)
{
	if (FN(isl_ast_node_check,NODE_TYPE)(node) < 0 || !field)
		goto error;
	if (node->FIELD == field) {
		FN(FIELD_TYPE,free)(field);
		return node;
	}

	node = isl_ast_node_cow(node);
	if (!node)
		goto error;

	FN(FIELD_TYPE,free)(node->FIELD);
	node->FIELD = field;

	return node;
error:
	isl_ast_node_free(node);
	FN(FIELD_TYPE,free)(field);
	return NULL;
}
