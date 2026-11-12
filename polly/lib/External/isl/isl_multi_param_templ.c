/*
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege
 */

#include <isl_multi_macro.h>

/* Does the multiple expression "multi" depend in any way
 * on the parameter with identifier "id"?
 */
isl_bool FN(MULTI(BASE),involves_param_id)(__isl_keep MULTI(BASE) *multi,
	__isl_keep isl_id *id)
{
	int i;
	int pos;

	if (!multi || !id)
		return isl_bool_error;
	if (multi->n == 0)
		return isl_bool_false;
	pos = FN(MULTI(BASE),find_dim_by_id)(multi, isl_dim_param, id);
	if (pos < 0)
		return isl_bool_false;

	for (i = 0; i < multi->n; ++i) {
		isl_bool involved = FN(EL,involves_param_id)(multi->u.p[i], id);
		if (involved < 0 || involved)
			return involved;
	}

	return isl_bool_false;
}

/* Does the multiple expression "multi" depend in any way
 * on any of the parameters with identifiers in "list"?
 */
isl_bool FN(MULTI(BASE),involves_param_id_list)(__isl_keep MULTI(BASE) *multi,
	__isl_keep isl_id_list *list)
{
	int i;
	isl_size n;

	n = isl_id_list_size(list);
	if (n < 0)
		return isl_bool_error;
	for (i = 0; i < n; ++i) {
		isl_bool involves;
		isl_id *id;

		id = isl_id_list_get_at(list, i);
		involves = FN(MULTI(BASE),involves_param_id)(multi, id);
		isl_id_free(id);

		if (involves < 0 || involves)
			return involves;
	}

	return isl_bool_false;
}
