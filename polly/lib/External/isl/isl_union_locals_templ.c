/*
 * Copyright 2020      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

/* isl_union_*_every_* callback that checks whether "pw"
 * is free of local variables.
 */
static isl_bool FN(UNION,no_locals_el)(__isl_keep PW *pw, void *user)
{
	return isl_bool_not(FN(PW,involves_locals)(pw));
}

/* Does "u" involve any local variables, i.e., integer divisions?
 */
isl_bool FN(UNION,involves_locals)(__isl_keep UNION *u)
{
	isl_bool no_locals;

	no_locals = FN(FN(UNION,every),BASE)(u, &FN(UNION,no_locals_el), NULL);

	return isl_bool_not(no_locals);
}
