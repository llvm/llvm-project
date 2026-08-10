/*
 * Copyright 2008      Katholieke Universiteit Leuven
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 */

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,TYPE_BASE)

/* Read an object of type TYPE from "str" (using an isl_stream).
 */
__isl_give TYPE *FN(isl,FN(TYPE_BASE,read_from_str))(isl_ctx *ctx,
	const char *str)
{
	TYPE *obj;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	obj = FN(isl_stream_read,TYPE_BASE)(s);
	isl_stream_free(s);
	return obj;
}
