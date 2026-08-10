/*
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* If "obj" involves a parameter with identifier "id",
 * then turn it into an existentially quantified variable.
 */
__isl_give TYPE *FN(TYPE,project_out_param_id)(__isl_take TYPE *obj,
	__isl_take isl_id *id)
{
	int pos;

	if (!obj || !id)
		goto error;
	pos = FN(TYPE,find_dim_by_id)(obj, isl_dim_param, id);
	isl_id_free(id);
	if (pos < 0)
		return obj;
	return FN(TYPE,project_out)(obj, isl_dim_param, pos, 1);
error:
	FN(TYPE,free)(obj);
	isl_id_free(id);
	return NULL;
}

/* If "obj" involves any of the parameters with identifiers in "list",
 * then turn them into existentially quantified variables.
 */
__isl_give TYPE *FN(TYPE,project_out_param_id_list)(__isl_take TYPE *obj,
	__isl_take isl_id_list *list)
{
	int i;
	isl_size n;

	n = isl_id_list_size(list);
	if (n < 0)
		goto error;
	for (i = 0; i < n; ++i) {
		isl_id *id;

		id = isl_id_list_get_at(list, i);
		obj = FN(TYPE,project_out_param_id)(obj, id);
	}

	isl_id_list_free(list);
	return obj;
error:
	isl_id_list_free(list);
	FN(TYPE,free)(obj);
	return NULL;
}
