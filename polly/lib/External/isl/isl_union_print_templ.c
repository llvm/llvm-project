/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#include "isl_union_macro.h"

/* Print "pw" in a sequence of "PART" objects delimited by semicolons.
 * Each "PART" object itself is also printed as a semicolon delimited
 * sequence of pieces.
 * If data->first = 1, then this is the first in the sequence.
 * Update data->first to tell the next element that it is not the first.
 */
static isl_stat FN(print_body_wrap,BASE)(__isl_take PART *pw,
	void *user)
{
	struct isl_union_print_data *data;
	data = (struct isl_union_print_data *) user;

	if (!data->first)
		data->p = isl_printer_print_str(data->p, "; ");
	data->first = 0;

	data->p = FN(print_body,BASE)(data->p, pw);
	FN(PART,free)(pw);

	return isl_stat_non_null(data->p);
}

/* Print the body of "u" (everything except the parameter declarations)
 * to "p" in isl format.
 */
static __isl_give isl_printer *FN(print_body_union,BASE)(
	__isl_take isl_printer *p, __isl_keep UNION *u)
{
	struct isl_union_print_data data;

	p = isl_printer_print_str(p, s_open_set[0]);
	data.p = p;
	data.first = 1;
	if (FN(FN(UNION,foreach),BASE)(u, &FN(print_body_wrap,BASE), &data) < 0)
		data.p = isl_printer_free(data.p);
	p = data.p;
	p = isl_printer_print_str(p, s_close_set[0]);

	return p;
}

/* Print the "UNION" object "u" to "p" in isl format.
 */
static __isl_give isl_printer *FN(FN(print_union,BASE),isl)(
	__isl_take isl_printer *p, __isl_keep UNION *u)
{
	struct isl_print_space_data space_data = { 0 };
	isl_space *space;

	space = FN(UNION,get_space)(u);
	p = print_param_tuple(p, space, &space_data);
	isl_space_free(space);

	p = FN(print_body_union,BASE)(p, u);

	return p;
}
