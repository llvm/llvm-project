#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Check that "obj" has a single reference.
 * That is, check that "obj" can be changed inplace.
 */
isl_stat FN(TYPE,check_single_reference)(__isl_keep TYPE *obj)
{
	isl_bool single;

	single = FN(TYPE,has_single_reference)(obj);
	if (single < 0)
		return isl_stat_error;
	if (!single)
		isl_die(FN(TYPE,get_ctx)(obj), isl_error_invalid,
			"object should have a single reference",
			return isl_stat_error);
	return isl_stat_ok;
}
