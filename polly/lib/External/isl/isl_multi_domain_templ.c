/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#include <isl/set.h>

#include <isl_multi_macro.h>

/* Return the shared domain of the elements of "multi".
 *
 * If "multi" has an explicit domain, then return this domain.
 */
__isl_give isl_set *FN(MULTI(BASE),domain)(__isl_take MULTI(BASE) *multi)
{
	int i;
	isl_set *dom;

	if (!multi)
		return NULL;

	if (FN(MULTI(BASE),has_explicit_domain)(multi)) {
		dom = FN(MULTI(BASE),get_explicit_domain)(multi);
		FN(MULTI(BASE),free)(multi);
		return dom;
	}

	dom = isl_set_universe(FN(MULTI(BASE),get_domain_space)(multi));
	for (i = 0; i < multi->n; ++i) {
		isl_set *dom_i;

		dom_i = FN(EL,domain)(FN(FN(MULTI(BASE),get),BASE)(multi, i));
		dom = isl_set_intersect(dom, dom_i);
	}

	FN(MULTI(BASE),free)(multi);
	return dom;
}
