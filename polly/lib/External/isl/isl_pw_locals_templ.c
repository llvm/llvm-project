/*
 * Copyright 2020      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl_pw_macro.h>

/* isl_pw_*_every_piece callback that checks whether "set" and "el"
 * are free of local variables.
 */
static isl_bool FN(PW,piece_no_local)(__isl_keep isl_set *set,
	__isl_keep EL *el, void *user)
{
	isl_bool involves;

	involves = isl_set_involves_locals(set);
	if (involves >= 0 && !involves)
		involves = FN(EL,involves_locals)(el);

	return isl_bool_not(involves);
}

/* Does "pw" involve any local variables, i.e., integer divisions?
 */
isl_bool FN(PW,involves_locals)(__isl_keep PW *pw)
{
	isl_bool no_locals;

	no_locals = FN(PW,every_piece)(pw, &FN(PW,piece_no_local), NULL);
	return isl_bool_not(no_locals);
}
