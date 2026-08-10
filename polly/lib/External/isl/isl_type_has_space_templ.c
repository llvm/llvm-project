/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Is the space of "obj" equal to "space"?
 */
isl_bool FN(TYPE,has_space)(__isl_keep TYPE *obj, __isl_keep isl_space *space)
{
	return isl_space_is_equal(FN(TYPE,peek_space)(obj), space);
}
