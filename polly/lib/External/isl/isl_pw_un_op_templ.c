/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, INRIA Saclay - Ile-de-France,
 * Parc Club Orsay Universite, ZAC des vignes, 4 rue Jacques Monod,
 * 91893 Orsay, France
 */

#include <isl_pw_macro.h>

/* Data structure that specifies how isl_pw_*_un_op should
 * modify its input.
 *
 * If "fn_space" is set, then it is applied to the space.
 *
 * If "fn_domain" is set, then it is applied to the cells.
 *
 * "fn_base" is applied to each base expression.
 * This function is assumed to have no effect on the default value
 * (i.e., zero for those objects with a default value).
 */
S(PW,un_op_control) {
	__isl_give isl_space *(*fn_space)(__isl_take isl_space *space);
	__isl_give isl_set *(*fn_domain)(__isl_take isl_set *domain);
	__isl_give EL *(*fn_base)(__isl_take EL *el);
};

/* Modify "pw" based on "control".
 *
 * If the cells are modified, then the corresponding base expressions
 * may need to be adjusted to the possibly modified equality constraints.
 */
static __isl_give PW *FN(PW,un_op)(__isl_take PW *pw,
	S(PW,un_op_control) *control)
{
	isl_space *space;
	isl_size n;
	int i;

	n = FN(PW,n_piece)(pw);
	if (n < 0)
		return FN(PW,free)(pw);

	for (i = n - 1; i >= 0; --i) {
		EL *el;
		isl_set *domain;

		el = FN(PW,take_base_at)(pw, i);
		el = control->fn_base(el);
		pw = FN(PW,restore_base_at)(pw, i, el);

		if (!control->fn_domain)
			continue;

		domain = FN(PW,take_domain_at)(pw, i);
		domain = control->fn_domain(domain);
		pw = FN(PW,restore_domain_at)(pw, i, domain);

		pw = FN(PW,exploit_equalities_and_remove_if_empty)(pw, i);
	}

	if (!control->fn_space)
		return pw;

	space = FN(PW,take_space)(pw);
	space = control->fn_space(space);
	pw = FN(PW,restore_space)(pw, space);

	return pw;
}
