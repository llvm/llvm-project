/*
 * Copyright 2011      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#undef TYPE
#define TYPE CAT(isl_pw_,BASE)

/* Read an object of type "TYPE" from "s" with parameter domain "dom".
 * "v" contains a description of the identifiers parsed so far.
 */
static __isl_give TYPE *FN(isl_stream_read_with_params_pw,BASE)(
	__isl_keep isl_stream *s, __isl_keep isl_set *dom, struct vars *v)
{
	TYPE *obj;

	obj = FN(read_conditional,BASE)(s, isl_set_copy(dom), v);

	while (isl_stream_eat_if_available(s, ';')) {
		TYPE *obj2;

		obj2 = FN(read_conditional,BASE)(s, isl_set_copy(dom), v);
		obj = FN(TYPE,union_add)(obj, obj2);
	}

	return obj;
}
