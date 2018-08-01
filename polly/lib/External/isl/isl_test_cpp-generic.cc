/* Copyright 2016-2017 Tobias Grosser
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Tobias Grosser, Weststrasse 47, CH-8003, Zurich
 */

#ifndef IS_TRUE
#define IS_TRUE(b)	(b)
#endif

/* Test the pointer interface for interaction between isl C and C++ types.
 *
 * This tests:
 * - construction from an isl C object
 * - check that constructed objects are non-null
 * - get a non-owned C pointer from an isl C++ object usable in __isl_keep
 *   methods
 * - use copy to get an owned C pointer from an isl C++ object which is usable
 *   in __isl_take methods. Verify that the original C++ object retains a valid
 *   pointer.
 * - use release to get an owned C pointer from an isl C++ object which is
 *   usable in __isl_take methods. Verify that the original C++ object gave up
 *   its pointer and now is null.
 */
void test_pointer(isl::ctx ctx)
{
	isl_set *c_empty = isl_set_read_from_str(ctx.get(), "{ : false }");
	isl::set empty = isl::manage(c_empty);
	assert(IS_TRUE(empty.is_empty()));
	assert(isl_set_is_empty(empty.get()));

	assert(!empty.is_null());
	isl_set_free(empty.copy());
	assert(!empty.is_null());
	isl_set_free(empty.release());
	assert(empty.is_null());
}

/* Test that isl objects can be constructed.
 *
 * This tests:
 *  - construction of a null object
 *  - construction from a string
 *  - construction from an integer
 *  - static constructor without a parameter
 *  - conversion construction (implicit)
 *  - conversion construction (explicit)
 *
 *  The tests to construct from integers and strings cover functionality that
 *  is also tested in the parameter type tests, but here we verify that
 *  multiple overloaded constructors are available and that overload resolution
 *  works as expected.
 *
 *  Construction from an isl C pointer is tested in test_pointer.
 */
void test_constructors(isl::ctx ctx)
{
	isl::val null;
	assert(null.is_null());

	isl::val zero_from_str = isl::val(ctx, "0");
	assert(IS_TRUE(zero_from_str.is_zero()));

	isl::val zero_int_con = isl::val(ctx, 0);
	assert(IS_TRUE(zero_int_con.is_zero()));

	isl::val zero_static_con = isl::val::zero(ctx);
	assert(IS_TRUE(zero_static_con.is_zero()));

	isl::basic_set bs(ctx, "{ [1] }");
	isl::set result(ctx, "{ [1] }");
	isl::set s = bs;
	assert(IS_TRUE(s.is_equal(result)));
	isl::set s2(bs);
	assert(IS_TRUE(s.unite(s2).is_equal(result)));
}

/* Test integer function parameters.
 *
 * Verify that extreme values and zero work.
 */
void test_parameters_int(isl::ctx ctx)
{
	isl::val long_max_str(ctx, std::to_string(LONG_MAX));
	isl::val long_max_int(ctx, LONG_MAX);
	assert(IS_TRUE(long_max_str.eq(long_max_int)));

	isl::val long_min_str(ctx, std::to_string(LONG_MIN));
	isl::val long_min_int(ctx, LONG_MIN);
	assert(IS_TRUE(long_min_str.eq(long_min_int)));

	isl::val long_zero_str = isl::val(ctx, std::to_string(0));
	isl::val long_zero_int = isl::val(ctx, 0);
	assert(IS_TRUE(long_zero_str.eq(long_zero_int)));
}

/* Test isl objects parameters.
 *
 * Verify that isl objects can be passed as lvalue and rvalue parameters.
 * Also verify that isl object parameters are automatically type converted if
 * there is an inheritance relation. Finally, test function calls without
 * any additional parameters, apart from the isl object on which
 * the method is called.
 */
void test_parameters_obj(isl::ctx ctx)
{
	isl::set a(ctx, "{ [0] }");
	isl::set b(ctx, "{ [1] }");
	isl::set c(ctx, "{ [2] }");
	isl::set expected(ctx, "{ [i] : 0 <= i <= 2 }");

	isl::set tmp = a.unite(b);
	isl::set res_lvalue_param = tmp.unite(c);
	assert(IS_TRUE(res_lvalue_param.is_equal(expected)));

	isl::set res_rvalue_param = a.unite(b).unite(c);
	assert(IS_TRUE(res_rvalue_param.is_equal(expected)));

	isl::basic_set a2(ctx, "{ [0] }");
	assert(IS_TRUE(a.is_equal(a2)));

	isl::val two(ctx, 2);
	isl::val half(ctx, "1/2");
	isl::val res_only_this_param = two.inv();
	assert(IS_TRUE(res_only_this_param.eq(half)));
}

/* Test different kinds of parameters to be passed to functions.
 *
 * This includes integer and isl C++ object parameters.
 */
void test_parameters(isl::ctx ctx)
{
	test_parameters_int(ctx);
	test_parameters_obj(ctx);
}

/* Test that isl objects are returned correctly.
 *
 * This only tests that after combining two objects, the result is successfully
 * returned.
 */
void test_return_obj(isl::ctx ctx)
{
	isl::val one(ctx, "1");
	isl::val two(ctx, "2");
	isl::val three(ctx, "3");

	isl::val res = one.add(two);

	assert(IS_TRUE(res.eq(three)));
}

/* Test that integer values are returned correctly.
 */
void test_return_int(isl::ctx ctx)
{
	isl::val one(ctx, "1");
	isl::val neg_one(ctx, "-1");
	isl::val zero(ctx, "0");

	assert(one.sgn() > 0);
	assert(neg_one.sgn() < 0);
	assert(zero.sgn() == 0);
}

/* Test that strings are returned correctly.
 * Do so by calling overloaded isl::ast_build::from_expr methods.
 */
void test_return_string(isl::ctx ctx)
{
	isl::set context(ctx, "[n] -> { : }");
	isl::ast_build build = isl::ast_build::from_context(context);
	isl::pw_aff pw_aff(ctx, "[n] -> { [n] }");
	isl::set set(ctx, "[n] -> { : n >= 0 }");

	isl::ast_expr expr = build.expr_from(pw_aff);
	const char *expected_string = "n";
	assert(expected_string == expr.to_C_str());

	expr = build.expr_from(set);
	expected_string = "n >= 0";
	assert(expected_string == expr.to_C_str());
}
