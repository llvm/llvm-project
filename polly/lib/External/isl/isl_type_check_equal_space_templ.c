/*
 * Copyright 2011      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

/* Check that "obj1" and "obj2" live in the same space,
 * reporting an error if they do not.
 */
isl_stat FN(TYPE_PAIR,check_equal_space)(__isl_keep TYPE1 *obj1,
	__isl_keep TYPE2 *obj2)
{
	isl_bool equal;

	equal = FN(TYPE_PAIR,has_equal_space)(obj1, obj2);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(FN(TYPE1,get_ctx)(obj1), isl_error_invalid,
			"spaces don't match", return isl_stat_error);

	return isl_stat_ok;
}
