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

__isl_give PW *FN(PW,sub)(__isl_take PW *pw1, __isl_take PW *pw2)
{
	return FN(PW,add)(pw1, FN(PW,neg)(pw2));
}
