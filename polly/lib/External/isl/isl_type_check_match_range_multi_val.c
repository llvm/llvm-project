#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Does the range space of "obj" match the space of "mv" (ignoring parameters)?
 */
static isl_bool FN(TYPE,match_range_multi_val)(__isl_keep TYPE *obj,
	__isl_keep isl_multi_val *mv)
{
	isl_space *space, *mv_space;

	space = FN(TYPE,peek_space)(obj);
	mv_space = isl_multi_val_peek_space(mv);
	return isl_space_tuple_is_equal(space, isl_dim_out,
					mv_space, isl_dim_set);
}

/* Check that the range space of "obj" matches the space of "mv"
 * (ignoring parameters).
 */
static isl_stat FN(TYPE,check_match_range_multi_val)(__isl_keep TYPE *obj,
	__isl_keep isl_multi_val *mv)
{
	isl_bool equal;

	equal = FN(TYPE,match_range_multi_val)(obj, mv);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(isl_multi_val_get_ctx(mv), isl_error_invalid,
			"spaces don't match", return isl_stat_error);
	return isl_stat_ok;
}
