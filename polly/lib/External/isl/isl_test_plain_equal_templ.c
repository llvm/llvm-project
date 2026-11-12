/*
 * Copyright 2012      Ecole Normale Superieure
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 */

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Is "obj" obviously equal to the object represented by "str"?
 */
static isl_bool FN(BASE,plain_is_equal)(__isl_keep TYPE *obj, const char *str)
{
	isl_ctx *ctx;
	TYPE *obj2;
	isl_bool equal;

	if (!obj)
		return isl_bool_error;

	ctx = FN(TYPE,get_ctx)(obj);
	obj2 = FN(TYPE,read_from_str)(ctx, str);
	equal = FN(TYPE,plain_is_equal)(obj, obj2);
	FN(TYPE,free)(obj2);

	return equal;
}

/* Check that "obj" is obviously equal to the object represented by "str".
 */
static isl_stat FN(BASE,check_plain_equal)(__isl_keep TYPE *obj,
	const char *str)
{
	isl_bool equal;

	equal = FN(BASE,plain_is_equal)(obj, str);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(FN(TYPE,get_ctx)(obj), isl_error_unknown,
			"result not as expected", return isl_stat_error);
	return isl_stat_ok;
}
