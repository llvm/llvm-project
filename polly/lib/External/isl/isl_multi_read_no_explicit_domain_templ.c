/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/space.h>
#include <isl/set.h>

#include <isl_multi_macro.h>

/* This function is called for each element in a tuple inside
 * isl_stream_read_multi_*.
 * Read an EL from "s" and add it to *list.
 */
static __isl_give isl_space *FN(read_el,BASE)(__isl_keep isl_stream *s,
	struct vars *v, __isl_take isl_space *space, int rational, void *user)
{
	LIST(EL) **list = (LIST(EL) **) user;
	EL *el;

	el = FN(isl_stream_read,BASE)(s);
	*list = FN(LIST(EL),add)(*list, el);
	if (!*list)
		return isl_space_free(space);

	return space;
}

/* Read a multi expression from "s".
 *
 * We first read a tuple space, collecting the element values in a list.
 * Then we create an isl_multi_* from the space and the isl_*_list.
 */
__isl_give MULTI(BASE) *FN(isl_stream_read_multi,BASE)(
	__isl_keep isl_stream *s)
{
	struct vars *v;
	isl_set *dom = NULL;
	isl_space *space;
	MULTI(BASE) *multi = NULL;
	LIST(EL) *list;

	v = vars_new(s->ctx);
	if (!v)
		return NULL;

	dom = isl_set_universe(isl_space_params_alloc(s->ctx, 0));
	if (next_is_tuple(s)) {
		dom = read_map_tuple(s, dom, isl_dim_param, v, 1, 0);
		if (isl_stream_eat(s, ISL_TOKEN_TO))
			goto error;
	}
	if (!isl_set_plain_is_universe(dom))
		isl_die(s->ctx, isl_error_invalid,
			"expecting universe parameter domain", goto error);
	if (isl_stream_eat(s, '{'))
		goto error;

	space = isl_set_get_space(dom);

	list = FN(LIST(EL),alloc)(s->ctx, 0);
	space = read_tuple_space(s, v, space, 1, 0, &FN(read_el,BASE), &list);
	multi = FN(FN(MULTI(BASE),from),LIST(BASE))(space, list);

	if (isl_stream_eat(s, '}'))
		goto error;

	vars_free(v);
	isl_set_free(dom);
	return multi;
error:
	vars_free(v);
	isl_set_free(dom);
	FN(MULTI(BASE),free)(multi);
	return NULL;
}

/* Read a multi expression from "str".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),read_from_str)(isl_ctx *ctx,
	const char *str)
{
	MULTI(BASE) *multi;
	isl_stream *s = isl_stream_new_str(ctx, str);
	if (!s)
		return NULL;
	multi = FN(isl_stream_read_multi,BASE)(s);
	isl_stream_free(s);
	return multi;
}
