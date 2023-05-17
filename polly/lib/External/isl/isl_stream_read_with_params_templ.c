/*
 * Copyright 2011      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,TYPE_BASE)

/* Read an object of type "TYPE" from "s".
 *
 * In particular, first read the parameters and the opening brace.
 * Then read the body that is specific to the object type.
 * Finally, read the closing brace.
 */
__isl_give TYPE *FN(isl_stream_read,TYPE_BASE)(__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom;
	TYPE *obj = NULL;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (isl_stream_eat(s, '{'))
		goto error;

	obj = FN(isl_stream_read_with_params,TYPE_BASE)(s, dom, v);

	if (isl_stream_eat(s, '}'))
		goto error;

	vars_free(v);
	isl_set_free(dom);
	return obj;
error:
	vars_free(v);
	isl_set_free(dom);
	FN(TYPE,free)(obj);
	return NULL;
}
