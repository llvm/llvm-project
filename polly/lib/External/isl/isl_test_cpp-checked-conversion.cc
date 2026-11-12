/*
 * Copyright 2018      Sven Verdoolaege
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege.
 */

#include <stdlib.h>

#include <isl/ctx.h>
#include <isl/options.h>
#include <isl/cpp-checked-conversion.h>

/* Check that converting a NULL object from the checked C++ bindings
 * (where the user is expected to check for NULL return values)
 * to the default C++ bindings (where exceptions are raised
 * instead of returning a NULL object) raises an exception.
 */
static void check_conversion_null(isl_ctx *ctx)
{
	isl::checked::set checked_set;
	isl::set set;

	bool caught = false;
	try {
		set = isl::uncheck(checked_set);
		isl_die(ctx, isl_error_unknown, "no exception raised", return);
	} catch (const isl::exception &e) {
		caught = true;
	}
	if (!caught)
		isl_die(ctx, isl_error_unknown, "no exception raised", return);
}

/* Dummy function on a set in the checked C++ bindings.
 */
static void f_checked(isl::checked::set set)
{
}

/* Dummy function on a set in the default C++ bindings.
 */
static void f_unchecked(isl::set set)
{
}

/* Check the conversion between C++ bindings in function calls.
 * An incorrect call will result in a compiler error.
 */
static void check_conversion_call(isl_ctx *ctx)
{
	isl::set set(ctx, "{ S[i] : 0 <= i < 10 }");
	isl::checked::set checked_set(ctx, "{ S[i] : 0 <= i < 10 }");

	f_unchecked(set);
	f_checked(isl::check(set));
	f_unchecked(isl::uncheck(checked_set));
	f_checked(checked_set);
}

/* Check that a double conversion results in the original set,
 * or at least something that is equal to the original set.
 */
static void check_conversion_equal(isl_ctx *ctx)
{
	isl::set set(ctx, "{ S[i] : 0 <= i < 10 }");
	isl::set set2;
	isl::checked::set checked_set;

	checked_set = isl::check(set);
	set2 = isl::uncheck(checked_set);

	if (!set.is_equal(set2))
		isl_die(ctx, isl_error_unknown, "bad conversion", return);
}

/* Perform some tests on the conversion between the default C++ bindings and
 * the checked C++ bindings.
 */
static void check_conversion(isl_ctx *ctx)
{
	check_conversion_null(ctx);
	check_conversion_call(ctx);
	check_conversion_equal(ctx);
}

int main()
{
	isl_ctx *ctx = isl_ctx_alloc();

	isl_options_set_on_error(ctx, ISL_ON_ERROR_ABORT);

	check_conversion(ctx);

	isl_ctx_free(ctx);

	return EXIT_SUCCESS;
}
