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

static isl_stat foreach_lifted_subset(__isl_take isl_set *set,
	__isl_take EL *el,
	isl_stat (*fn)(__isl_take isl_set *set, __isl_take EL *el,
		void *user), void *user)
{
	int i;

	if (!set || !el)
		goto error;

	for (i = 0; i < set->n; ++i) {
		isl_set *lift;
		EL *copy;

		lift = isl_set_from_basic_set(isl_basic_set_copy(set->p[i]));
		lift = isl_set_lift(lift);

		copy = FN(EL,copy)(el);
		copy = FN(EL,lift)(copy, isl_set_get_space(lift));

		if (fn(lift, copy, user) < 0)
			goto error;
	}

	isl_set_free(set);
	FN(EL,free)(el);

	return isl_stat_ok;
error:
	isl_set_free(set);
	FN(EL,free)(el);
	return isl_stat_error;
}

isl_stat FN(PW,foreach_lifted_piece)(__isl_keep PW *pw,
	isl_stat (*fn)(__isl_take isl_set *set, __isl_take EL *el,
		    void *user), void *user)
{
	int i;

	if (!pw)
		return isl_stat_error;

	for (i = 0; i < pw->n; ++i) {
		isl_bool any;
		isl_set *set;
		EL *el;

		any = isl_set_involves_locals(pw->p[i].set);
		if (any < 0)
			return isl_stat_error;
		set = isl_set_copy(pw->p[i].set);
		el = FN(EL,copy)(pw->p[i].FIELD);
		if (!any) {
			if (fn(set, el, user) < 0)
				return isl_stat_error;
			continue;
		}
		if (foreach_lifted_subset(set, el, fn, user) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}
