/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl/space.h>
#include <isl/set.h>

/* Given a piecewise function on a domain (A -> B),
 * interchange A and B in the wrapped domain
 * to obtain a function on the domain (B -> A).
 */
__isl_give PW *FN(PW,domain_reverse)(__isl_take PW *pw)
{
	S(PW,un_op_control) control = {
		.fn_space = &isl_space_domain_reverse,
		.fn_domain = &isl_set_wrapped_reverse,
		.fn_base = &FN(EL,domain_reverse),
	};
	return FN(PW,un_op)(pw, &control);
}
