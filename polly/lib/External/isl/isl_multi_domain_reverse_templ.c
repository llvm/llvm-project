/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl/space.h>

#include <isl_multi_macro.h>

/* Given a multi expression on a domain (A -> B),
 * interchange A and B in the wrapped domain
 * to obtain a multi expression on the domain (B -> A).
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),domain_reverse)(
	__isl_take MULTI(BASE) *multi)
{
	S(MULTI(BASE),un_op_control) control = {
		.fn_space = &isl_space_domain_reverse,
		.fn_el = &FN(EL,domain_reverse),
	};
	return FN(MULTI(BASE),un_op)(multi, &control);
}
