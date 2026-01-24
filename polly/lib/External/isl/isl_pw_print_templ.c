/*
 * Copyright 2011      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#undef EL
#define EL CAT(isl_,BASE)
#undef PW
#define PW CAT(isl_pw_,BASE)

/* Print the empty body of a piecewise expression.
 *
 * In particular, print the space with some arbitrary value (zero)
 * for all dimensions, followed by unsatisfiable constraints (false).
 */
static __isl_give isl_printer *FN(print_empty_body_pw,BASE)(
	__isl_take isl_printer *p, __isl_keep PW *pw)
{
	struct isl_print_space_data data = { 0 };
	isl_space *space;

	space = FN(PW,get_space)(pw);
	data.print_dim = &print_dim_zero;
	p = isl_print_space(space, p, 0, &data);
	isl_space_free(space);
	p = isl_printer_print_str(p, " : false");
	return p;
}

/* Print the body of a piecewise expression, i.e., a semicolon delimited
 * sequence of expressions, each followed by constraints.
 */
static __isl_give isl_printer *FN(print_body_pw,BASE)(
	__isl_take isl_printer *p, __isl_keep PW *pw)
{
	int i;
	isl_size n;

	n = FN(PW,n_piece)(pw);
	if (n < 0)
		return isl_printer_free(p);

	if (n == 0)
		FN(print_empty_body_pw,BASE)(p, pw);

	for (i = 0; i < n; ++i) {
		EL *el;
		isl_space *space;

		if (i)
			p = isl_printer_print_str(p, "; ");
		el = FN(PW,peek_base_at)(pw, i);
		p = FN(print_body,BASE)(p, el);
		space = FN(EL,get_domain_space)(el);
		p = print_disjuncts(set_to_map(pw->p[i].set), space, p, 0);
		isl_space_free(space);
	}
	return p;
}

/* Print a piecewise expression in isl format.
 */
static __isl_give isl_printer *FN(FN(print_pw,BASE),isl)(
	__isl_take isl_printer *p, __isl_keep PW *pw)
{
	struct isl_print_space_data data = { 0 };

	if (!pw)
		return isl_printer_free(p);

	p = print_param_tuple(p, pw->dim, &data);
	p = isl_printer_print_str(p, "{ ");
	p = FN(print_body_pw,BASE)(p, pw);
	p = isl_printer_print_str(p, " }");
	return p;
}
