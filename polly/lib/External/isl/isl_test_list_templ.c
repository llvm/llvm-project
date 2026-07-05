#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)
#define xLIST(EL) EL ## _list
#define LIST(EL) xLIST(EL)

#undef SET
#define SET	CAT(isl_,SET_BASE)
#undef EL
#define EL	CAT(isl_,EL_BASE)

/* Check that the conversion from SET to list of EL works as expected,
 * using input described by "str".
 */
static isl_stat FN(FN(FN(test_get_list,EL_BASE),from),SET_BASE)(isl_ctx *ctx,
	const char *str)
{
	int i;
	isl_size n;
	isl_bool equal;
	SET *set, *set2;
	LIST(EL) *list;

	set = FN(SET,read_from_str)(ctx, str);
	list = FN(FN(SET,get),LIST(EL_BASE))(set);

	set2 = FN(SET,empty)(FN(SET,get_space)(set));

	n = FN(LIST(EL),size)(list);
	for (i = 0; i < n; i++) {
		EL *el;
		el = FN(LIST(EL),get_at)(list, i);
		set2 = FN(SET,union)(set2, FN(FN(SET,from),EL_BASE)(el));
	}

	equal = FN(SET,is_equal)(set, set2);

	FN(SET,free)(set);
	FN(SET,free)(set2);
	FN(LIST(EL),free)(list);

	if (n < 0 || equal < 0)
		return isl_stat_error;

	if (!equal)
		isl_die(ctx, isl_error_unknown, "collections are not equal",
			return isl_stat_error);

	return isl_stat_ok;
}
