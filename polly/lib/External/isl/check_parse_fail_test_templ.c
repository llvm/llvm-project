/*
 * Copyright 2021      Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege,
 * Cerebras Systems, 1237 E Arques Ave, Sunnyvale, CA, USA
 */

#define xCAT(A,B) A ## B
#define CAT(A,B) xCAT(A,B)
#undef TYPE
#define TYPE CAT(isl_,BASE)
#define xFN(TYPE,NAME) TYPE ## _ ## NAME
#define FN(TYPE,NAME) xFN(TYPE,NAME)

#undef TESTS
#define TESTS CAT(parse_,CAT(BASE,_fail_tests))

/* Test parsing of objects of type TYPE
 * that are expected to fail.
 */
static isl_stat FN(check,TESTS)(isl_ctx *ctx)
{
	int i, n;
	int on_error;

	on_error = isl_options_get_on_error(ctx);
	isl_options_set_on_error(ctx, ISL_ON_ERROR_CONTINUE);
	n = ARRAY_SIZE(TESTS);
	for (i = 0; i < n; ++i) {
		TYPE *obj;

		obj = FN(TYPE,read_from_str)(ctx, TESTS[i]);
		FN(TYPE,free)(obj);
		if (obj)
			break;
	}
	isl_options_set_on_error(ctx, on_error);
	if (i < n)
		isl_die(ctx, isl_error_unknown,
			"parsing not expected to succeed",
			return isl_stat_error);

	return isl_stat_ok;
}
