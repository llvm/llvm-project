/*
 * Copyright 2018      Sven Verdoolaege
 * Copyright 2019      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 175 S San Antonio Rd, Los Altos, CA, USA
 */

/* Does the (range) tuple of "pw" have an identifier?
 *
 * Technically, the implementation should use isl_dim_set if "pw"
 * lives in a set space and isl_dim_out if it lives in a map space.
 * Internally, however, it can be assumed that isl_dim_set is equal
 * to isl_dim_out.
 */
isl_bool FN(PW,has_range_tuple_id)(__isl_keep PW *pw)
{
	return FN(PW,has_tuple_id)(pw, isl_dim_out);
}

/* Return the identifier of the (range) tuple of "pw", assuming it has one.
 *
 * Technically, the implementation should use isl_dim_set if "pw"
 * lives in a set space and isl_dim_out if it lives in a map space.
 * Internally, however, it can be assumed that isl_dim_set is equal
 * to isl_dim_out.
 */
__isl_give isl_id *FN(PW,get_range_tuple_id)(__isl_keep PW *pw)
{
	return FN(PW,get_tuple_id)(pw, isl_dim_out);
}

/* Replace the identifier of the (range) tuple of "pw" by "id".
 *
 * Technically, the implementation should use isl_dim_set if "pw"
 * lives in a set space and isl_dim_out if it lives in a map space.
 * Internally, however, it can be assumed that isl_dim_set is equal
 * to isl_dim_out.
 */
__isl_give PW *FN(PW,set_range_tuple_id)(__isl_take PW *pw,
	__isl_take isl_id *id)
{
	return FN(PW,set_tuple_id)(pw, isl_dim_out, id);
}
