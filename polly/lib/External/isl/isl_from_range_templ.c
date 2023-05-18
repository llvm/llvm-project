/*
 * Copyright 2013      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

/* Convert an object defined over a parameter domain
 * into one that is defined over a zero-dimensional set.
 */
__isl_give TYPE *FN(TYPE,from_range)(__isl_take TYPE *obj)
{
	isl_space *space;

	if (!obj)
		return NULL;
	if (!isl_space_is_set(FN(TYPE,peek_space)(obj)))
		isl_die(FN(TYPE,get_ctx)(obj), isl_error_invalid,
			"not living in a set space",
			return FN(TYPE,free)(obj));

	space = FN(TYPE,get_space)(obj);
	space = isl_space_from_range(space);
	obj = FN(TYPE,reset_space)(obj, space);

	return obj;
}
