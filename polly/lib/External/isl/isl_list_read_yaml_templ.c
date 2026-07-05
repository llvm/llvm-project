/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/stream.h>

#include <isl_list_macro.h>

/* Read a sequence of EL objects and return them as a list.
 */
static __isl_give LIST(EL) *FN(isl_stream_yaml_read,LIST(EL_BASE))(
	isl_stream *s)
{
	isl_ctx *ctx;
	LIST(EL) *list;
	isl_bool more;

	ctx = isl_stream_get_ctx(s);

	if (isl_stream_yaml_read_start_sequence(s) < 0)
		return NULL;

	list = FN(LIST(EL),alloc)(ctx, 0);
	while ((more = isl_stream_yaml_next(s)) == isl_bool_true) {
		EL *el;

		el = FN(isl_stream_read,EL_BASE)(s);
		list = FN(LIST(EL),add)(list, el);
	}

	if (more < 0 || isl_stream_yaml_read_end_sequence(s) < 0)
		return FN(LIST(EL),free)(list);

	return list;
}
