/*
 * Use of this software is governed by the MIT license
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Drop all parameters not referenced by "obj".
 */
__isl_give TYPE *FN(TYPE,drop_unused_params)(__isl_take TYPE *obj)
{
	int i;
	isl_size n;

	n = FN(TYPE,dim)(obj, isl_dim_param);
	if (n < 0 || FN(TYPE,check_named_params)(obj) < 0)
		return FN(TYPE,free)(obj);

	for (i = n - 1; i >= 0; i--) {
		isl_bool involves;

		involves = FN(TYPE,involves_dims)(obj, isl_dim_param, i, 1);
		if (involves < 0)
			return FN(TYPE,free)(obj);
		if (!involves)
			obj = FN(TYPE,drop_dims)(obj, isl_dim_param, i, 1);
	}

	return obj;
}
