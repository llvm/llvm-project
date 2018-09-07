/* Copyright 2016-2017 Tobias Grosser
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Tobias Grosser, Weststrasse 47, CH-8003, Zurich
 */

#include <vector>
#include <string>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <isl/options.h>
#include <isl/cpp.h>

static void die_impl(const char *file, int line, const char *message)
{
	fprintf(stderr, "Assertion failed in %s:%d %s\n", file, line, message);
	exit(EXIT_FAILURE);
}

static void assert_impl(bool condition, const char *file, int line,
	const char *message)
{
	if (condition)
		return;

	return die_impl(file, line, message);
}

#define die(msg) die_impl(__FILE__, __LINE__, msg)
#define assert(exp) assert_impl(exp, __FILE__, __LINE__, #exp)

#include "isl_test_cpp-generic.cc"

/* Test that isl_bool values are returned correctly.
 *
 * In particular, check the conversion to bool in case of true and false, and
 * exception throwing in case of error.
 */
static void test_return_bool(isl::ctx ctx)
{
	isl::set empty(ctx, "{ : false }");
	isl::set univ(ctx, "{ : }");
	isl::set null;

	bool b_true = empty.is_empty();
	bool b_false = univ.is_empty();
	bool caught = false;
	try {
		null.is_empty();
		die("no exception raised");
	} catch (const isl::exception_invalid &e) {
		caught = true;
	}

	assert(b_true);
	assert(!b_false);
	assert(caught);
}

/* Test that return values are handled correctly.
 *
 * Test that isl C++ objects, integers, boolean values, and strings are
 * returned correctly.
 */
static void test_return(isl::ctx ctx)
{
	test_return_obj(ctx);
	test_return_int(ctx);
	test_return_bool(ctx);
	test_return_string(ctx);
}

/* Test that foreach functions are modeled correctly.
 *
 * Verify that lambdas are correctly called as callback of a 'foreach'
 * function and that variables captured by the lambda work correctly. Also
 * check that the foreach function handles exceptions thrown from
 * the lambda and that it propagates the exception.
 */
static void test_foreach(isl::ctx ctx)
{
	isl::set s(ctx, "{ [0]; [1]; [2] }");

	std::vector<isl::basic_set> basic_sets;

	auto add_to_vector = [&] (isl::basic_set bs) {
		basic_sets.push_back(bs);
	};

	s.foreach_basic_set(add_to_vector);

	assert(basic_sets.size() == 3);
	assert(isl::set(basic_sets[0]).is_subset(s));
	assert(isl::set(basic_sets[1]).is_subset(s));
	assert(isl::set(basic_sets[2]).is_subset(s));
	assert(!basic_sets[0].is_equal(basic_sets[1]));

	auto fail = [&] (isl::basic_set bs) {
		throw "fail";
	};

	bool caught = false;
	try {
		s.foreach_basic_set(fail);
		die("no exception raised");
	} catch (char const *s) {
		caught = true;
	}
	assert(caught);
}

/* Test that an exception is generated for an isl error and
 * that the error message is captured by the exception.
 * Also check that the exception can be copied and that copying
 * does not throw any exceptions.
 */
static void test_exception(isl::ctx ctx)
{
	isl::multi_union_pw_aff mupa(ctx, "[]");
	isl::exception copy;

	static_assert(std::is_nothrow_copy_constructible<isl::exception>::value,
		"exceptions must be nothrow-copy-constructible");
	static_assert(std::is_nothrow_assignable<isl::exception,
						isl::exception>::value,
		"exceptions must be nothrow-assignable");

	try {
		auto umap = isl::union_map::from(mupa);
	} catch (const isl::exception_unsupported &error) {
		die("caught wrong exception");
	} catch (const isl::exception &error) {
		assert(strstr(error.what(), "without explicit domain"));
		copy = error;
	}
	assert(strstr(copy.what(), "without explicit domain"));
}

/* Test the (unchecked) isl C++ interface
 *
 * This includes:
 *  - The isl C <-> C++ pointer interface
 *  - Object construction
 *  - Different parameter types
 *  - Different return types
 *  - Foreach functions
 *  - Exceptions
 */
int main()
{
	isl_ctx *ctx = isl_ctx_alloc();

	isl_options_set_on_error(ctx, ISL_ON_ERROR_ABORT);

	test_pointer(ctx);
	test_constructors(ctx);
	test_parameters(ctx);
	test_return(ctx);
	test_foreach(ctx);
	test_exception(ctx);

	isl_ctx_free(ctx);

	return EXIT_SUCCESS;
}
