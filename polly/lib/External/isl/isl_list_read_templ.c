/*
 * Copyright 2017      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include <isl/stream.h>

#include <isl_list_macro.h>

/* Read a list of elements of type EL from "s".
 * The input format corresponds to the way lists are printed
 * by isl_printer_print_list_*.
 * In particular, the elements are separated by a comma and
 * the entire list is surrounded by parentheses.
 */
__isl_give LIST(EL) *FN(isl_stream_read,LIST(EL_BASE))(isl_stream *s)
{
	isl_ctx *ctx;
	LIST(EL) *list;

	if (!s)
		return NULL;
	ctx = isl_stream_get_ctx(s);
	list = FN(LIST(EL),alloc)(ctx, 0);
	if (!list)
		return NULL;
	if (isl_stream_eat(s, '(') < 0)
		return FN(LIST(EL),free)(list);
	if (isl_stream_eat_if_available(s, ')'))
		return list;
	do {
		EL *el;

		el = FN(isl_stream_read,EL_BASE)(s);
		list = FN(LIST(EL),add)(list, el);
		if (!list)
			return NULL;
	} while (isl_stream_eat_if_available(s, ','));
	if (isl_stream_eat(s, ')') < 0)
		return FN(LIST(EL),free)(list);
	return list;
}

#undef TYPE_BASE
#define TYPE_BASE	LIST(EL_BASE)
#include "isl_read_from_str_templ.c"
