#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

#undef TESTS
#define TESTS CAT(reparse_,CAT(BASE,_tests))

/* Test parsing of objects of type TYPE by printing
 * the expressions and checking that parsing the output results
 * in the same expression.
 * Do this for a set of expressions parsed from strings.
 */
static isl_stat FN(check,TESTS)(isl_ctx *ctx)
{
	int i;

	for (i = 0; i < ARRAY_SIZE(TESTS); ++i) {
		TYPE *obj;

		obj = FN(TYPE,read_from_str)(ctx, TESTS[i]);
		if (FN(check_reparse,BASE)(ctx, obj) < 0)
			return isl_stat_error;
	}

	return isl_stat_ok;
}
