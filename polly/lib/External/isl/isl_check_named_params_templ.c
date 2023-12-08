#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Check that "obj" has only named parameters, reporting an error
 * if it does not.
 */
isl_stat FN(TYPE,check_named_params)(__isl_keep TYPE *obj)
{
	return isl_space_check_named_params(FN(TYPE,peek_space)(obj));
}
