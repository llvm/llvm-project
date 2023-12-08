/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

/* Return the (elementwise) minimum of "multi1" and "multi2".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),min)(__isl_take MULTI(BASE) *multi1,
	__isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),bin_op)(multi1, multi2, &FN(EL,min));
}

/* Return the (elementwise) maximum of "multi1" and "multi2".
 */
__isl_give MULTI(BASE) *FN(MULTI(BASE),max)(__isl_take MULTI(BASE) *multi1,
	__isl_take MULTI(BASE) *multi2)
{
	return FN(MULTI(BASE),bin_op)(multi1, multi2, &FN(EL,max));
}
