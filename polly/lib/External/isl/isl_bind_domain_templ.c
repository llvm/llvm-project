/*
 * Copyright 2018      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

#include <isl/space.h>

/* Merge parameter "param" into the input dimension "i" of "obj".
 *
 * First plug in the parameter for the input dimension in "obj".
 * The drop the (now defunct) input dimension and
 * move the parameter in its original position.
 * Since dimension manipulations destroy spaces, modify the space
 * separately by only dropping the parameter.
 */
static __isl_give TYPE *FN(TYPE,merge_param)(__isl_take TYPE *obj, int i,
	int param)
{
	isl_id *id;
	isl_aff *aff;
	isl_space *space;
	isl_multi_aff *ma;

	space = FN(TYPE,get_domain_space)(obj);
	id = isl_space_get_dim_id(space, isl_dim_param, param);
	aff = isl_aff_param_on_domain_space_id(isl_space_copy(space), id);
	space = isl_space_map_from_set(space);
	ma = isl_multi_aff_identity(space);
	ma = isl_multi_aff_set_aff(ma, i, aff);
	obj = FN(TYPE,pullback_multi_aff)(obj, ma);
	space = FN(TYPE,get_domain_space)(obj);
	obj = FN(TYPE,drop_dims)(obj, isl_dim_in, i, 1);
	obj = FN(TYPE,move_dims)(obj, isl_dim_in, i, isl_dim_param, param, 1);
	space = isl_space_drop_dims(space, isl_dim_param, param, 1);
	obj = FN(TYPE,reset_domain_space)(obj, space);

	return obj;
}

/* Given a tuple of identifiers "tuple" that correspond
 * to the initial input dimensions of "obj",
 * if any of those identifiers appear as parameters
 * in "obj", then equate those parameters with the corresponding
 * input dimensions and project out the parameters.
 * The result therefore has no such parameters.
 */
static __isl_give TYPE *FN(TYPE,equate_initial_params)(__isl_take TYPE *obj,
	__isl_keep isl_multi_id *tuple)
{
	int i;
	isl_size n;

	n = isl_multi_id_size(tuple);
	if (n < 0)
		return FN(TYPE,free)(obj);
	for (i = 0; i < n; ++i) {
		isl_id *id;
		int pos;

		id = isl_multi_id_get_at(tuple, i);
		if (!id)
			return FN(TYPE,free)(obj);
		pos = FN(TYPE,find_dim_by_id)(obj, isl_dim_param, id);
		isl_id_free(id);
		if (pos < 0)
			continue;
		obj = FN(TYPE,merge_param)(obj, i, pos);
	}

	return obj;
}

/* Given a tuple of identifiers "tuple" in a space that corresponds
 * to the domain of "obj", if any of those identifiers appear as parameters
 * in "obj", then equate those parameters with the corresponding
 * input dimensions and project out the parameters.
 * The result therefore has no such parameters.
 */
static __isl_give TYPE *FN(TYPE,equate_domain_params)(__isl_take TYPE *obj,
	__isl_keep isl_multi_id *tuple)
{
	isl_stat r;
	isl_space *obj_space, *tuple_space;

	obj_space = FN(TYPE,get_space)(obj);
	tuple_space = isl_multi_id_peek_space(tuple);
	r = isl_space_check_domain_tuples(tuple_space, obj_space);
	isl_space_free(obj_space);
	if (r < 0)
		return FN(TYPE,free)(obj);

	return FN(TYPE,equate_initial_params)(obj, tuple);
}

/* Bind the domain dimensions of the function "obj" to parameters
 * with identifiers specified by "tuple", living in the same space
 * as the domain of "obj".
 *
 * If no parameters with these identifiers appear in "obj" already,
 * then the domain dimensions are simply reinterpreted as parameters.
 * Otherwise, the parameters are first equated to the corresponding
 * domain dimensions.
 */
__isl_give TYPE *FN(TYPE,bind_domain)(__isl_take TYPE *obj,
	__isl_take isl_multi_id *tuple)
{
	isl_space *space;

	obj = FN(TYPE,equate_domain_params)(obj, tuple);
	space = FN(TYPE,get_space)(obj);
	space = isl_space_bind_map_domain(space, tuple);
	isl_multi_id_free(tuple);
	obj = FN(TYPE,reset_space)(obj, space);

	return obj;
}

/* Given a tuple of identifiers "tuple" in a space that corresponds
 * to the domain of the wrapped relation in the domain of "obj",
 * if any of those identifiers appear as parameters
 * in "obj", then equate those parameters with the corresponding
 * input dimensions and project out the parameters.
 * The result therefore has no such parameters.
 */
static __isl_give TYPE *FN(TYPE,equate_domain_wrapped_domain_params)(
	__isl_take TYPE *obj, __isl_keep isl_multi_id *tuple)
{
	isl_stat r;
	isl_space *obj_space, *tuple_space;

	obj_space = FN(TYPE,get_space)(obj);
	tuple_space = isl_multi_id_peek_space(tuple);
	r = isl_space_check_domain_wrapped_domain_tuples(tuple_space,
							obj_space);
	isl_space_free(obj_space);
	if (r < 0)
		return FN(TYPE,free)(obj);

	return FN(TYPE,equate_initial_params)(obj, tuple);
}

/* Given a function living in a space of the form [A -> B] -> C and
 * a tuple of identifiers in A, bind the domain dimensions of the relation
 * wrapped in the domain of "obj" with identifiers specified by "tuple",
 * returning a function in the space B -> C.
 *
 * If no parameters with these identifiers appear in "obj" already,
 * then the domain dimensions are simply reinterpreted as parameters.
 * Otherwise, the parameters are first equated to the corresponding
 * domain dimensions.
 */
__isl_give TYPE *FN(TYPE,bind_domain_wrapped_domain)(__isl_take TYPE *obj,
	__isl_take isl_multi_id *tuple)
{
	isl_space *space;

	obj = FN(TYPE,equate_domain_wrapped_domain_params)(obj, tuple);
	space = FN(TYPE,get_space)(obj);
	space = isl_space_bind_domain_wrapped_domain(space, tuple);
	isl_multi_id_free(tuple);
	obj = FN(TYPE,reset_space)(obj, space);

	return obj;
}
