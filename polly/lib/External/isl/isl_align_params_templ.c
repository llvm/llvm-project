/*
 * Copyright 2011      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Align the parameters of the two arguments of type ARG1 and ARG2
 * (if needed).
 */
isl_stat FN(FN(ARG1,align_params),SUFFIX)(__isl_keep ARG1 **obj1,
	__isl_keep ARG2 **obj2)
{
	isl_space *space1, *space2;
	isl_bool equal_params;

	space1 = FN(ARG1,peek_space)(*obj1);
	space2 = FN(ARG2,peek_space)(*obj2);
	equal_params = isl_space_has_equal_params(space1, space2);
	if (equal_params < 0)
		goto error;
	if (equal_params)
		return isl_stat_ok;
	if (FN(ARG1,check_named_params)(*obj1) < 0 ||
	    FN(ARG2,check_named_params)(*obj2) < 0)
		goto error;
	*obj1 = FN(ARG1,align_params)(*obj1, FN(ARG2,get_space)(*obj2));
	*obj2 = FN(ARG2,align_params)(*obj2, FN(ARG1,get_space)(*obj1));
	if (!*obj1 || !*obj2)
		goto error;
	return isl_stat_ok;
error:
	*obj1 = FN(ARG1,free)(*obj1);
	*obj2 = FN(ARG2,free)(*obj2);
	return isl_stat_error;
}
