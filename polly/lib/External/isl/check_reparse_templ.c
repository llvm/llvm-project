#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

/* Check that printing "obj" and parsing the output results
 * in the same expression.
 */
static isl_stat FN(check_reparse,BASE)(isl_ctx *ctx,
	__isl_take TYPE *obj)
{
	char *str;
	isl_bool equal;
	TYPE *obj2;

	str = FN(TYPE,to_str)(obj);
	obj2 = FN(TYPE,read_from_str)(ctx, str);
	free(str);
	equal = FN(TYPE,plain_is_equal)(obj, obj2);
	FN(TYPE,free)(obj);
	FN(TYPE,free)(obj2);
	if (equal < 0)
		return isl_stat_error;
	if (!equal)
		isl_die(ctx, isl_error_unknown,
			"parsed function not equal to original",
			return isl_stat_error);

	return isl_stat_ok;
}
