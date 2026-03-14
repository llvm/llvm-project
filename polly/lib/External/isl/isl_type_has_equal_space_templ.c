/*
 * Copyright 2010      INRIA Saclay
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

#ifndef PEEK_SPACE
#define PEEK_SPACE	peek_space
#endif

/* Do "obj1" and "obj2" have the same space?
 */
isl_bool FN(TYPE_PAIR,has_equal_space)(__isl_keep TYPE1 *obj1,
	__isl_keep TYPE2 *obj2)
{
	isl_space *space1, *space2;

	space1 = FN(TYPE1,PEEK_SPACE)(obj1);
	space2 = FN(TYPE2,PEEK_SPACE)(obj2);
	return isl_space_is_equal(space1, space2);
}
