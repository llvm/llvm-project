#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Check that there are "n" dimensions of type "type" starting at "first"
 * in "obj".
 */
isl_stat FN(TYPE,check_range)(__isl_keep TYPE *obj,
	enum isl_dim_type type, unsigned first, unsigned n)
{
	isl_size dim;

	dim = FN(TYPE,dim)(obj, type);
	if (dim < 0)
		return isl_stat_error;
	if (first + n > dim || first + n < first)
		isl_die(FN(TYPE,get_ctx)(obj), isl_error_invalid,
			"position or range out of bounds",
			return isl_stat_error);
	return isl_stat_ok;
}
