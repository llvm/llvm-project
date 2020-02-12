#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Does "obj" have a single reference?
 * That is, can "obj" be changed inplace?
 */
isl_bool FN(TYPE,has_single_reference)(__isl_keep TYPE *obj)
{
	if (!obj)
		return isl_bool_error;
	return obj->ref == 1;
}
