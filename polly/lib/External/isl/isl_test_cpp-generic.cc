/* Copyright 2016-2017 Tobias Grosser
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Tobias Grosser, Weststrasse 47, CH-8003, Zurich
 */

#ifndef IS_TRUE
#define IS_TRUE(b)	(b)
#endif
#ifndef SIZE_VAL
#define SIZE_VAL(s)	(s)
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
 *  - construction of empty union set
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

	isl::union_set us(ctx, "{ A[1]; B[2, 3] }");
	isl::union_set empty = isl::union_set::empty(ctx);
	assert(IS_TRUE(us.is_equal(us.unite(empty))));
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

/* Test the functionality of "every" functions
 * that does not depend on the type of C++ bindings.
 */
static void test_every_generic(isl::ctx ctx)
{
	isl::union_set us(ctx, "{ A[i]; B[j] }");

	auto is_empty = [] (isl::set s) {
		return s.is_empty();
	};
	assert(!IS_TRUE(us.every_set(is_empty)));

	auto is_non_empty = [] (isl::set s) {
		return !s.is_empty();
	};
	assert(IS_TRUE(us.every_set(is_non_empty)));

	auto in_A = [] (isl::set s) {
		return s.is_subset(isl::set(s.ctx(), "{ A[x] }"));
	};
	assert(!IS_TRUE(us.every_set(in_A)));

	auto not_in_A = [] (isl::set s) {
		return !s.is_subset(isl::set(s.ctx(), "{ A[x] }"));
	};
	assert(!IS_TRUE(us.every_set(not_in_A)));
}

/* Check basic construction of spaces.
 */
static void test_space(isl::ctx ctx)
{
	isl::space unit = isl::space::unit(ctx);
	isl::space set_space = unit.add_named_tuple("A", 3);
	isl::space map_space = set_space.add_named_tuple("B", 2);

	isl::set set = isl::set::universe(set_space);
	isl::map map = isl::map::universe(map_space);
	assert(IS_TRUE(set.is_equal(isl::set(ctx, "{ A[*,*,*] }"))));
	assert(IS_TRUE(map.is_equal(isl::map(ctx, "{ A[*,*,*] -> B[*,*] }"))));
}

/* Construct a simple schedule tree with an outer sequence node and
 * a single-dimensional band node in each branch, with one of them
 * marked coincident.
 */
static isl::schedule construct_schedule_tree(isl::ctx ctx)
{
	isl::union_set A(ctx, "{ A[i] : 0 <= i < 10 }");
	isl::union_set B(ctx, "{ B[i] : 0 <= i < 20 }");

	auto node = isl::schedule_node::from_domain(A.unite(B));
	node = node.child(0);

	isl::union_set_list filters(ctx, 0);
	filters = filters.add(A).add(B);
	node = node.insert_sequence(filters);

	isl::multi_union_pw_aff f_A(ctx, "[ { A[i] -> [i] } ]");
	node = node.child(0);
	node = node.child(0);
	node = node.insert_partial_schedule(f_A);
	auto band = node.as<isl::schedule_node_band>();
	band = band.member_set_coincident(0, true);
	node = band.ancestor(2);

	isl::multi_union_pw_aff f_B(ctx, "[ { B[i] -> [i] } ]");
	node = node.child(1);
	node = node.child(0);
	node = node.insert_partial_schedule(f_B);
	node = node.ancestor(2);

	return node.schedule();
}

/* Test basic schedule tree functionality that is independent
 * of the type of bindings.
 *
 * In particular, create a simple schedule tree and
 * - check that the root node is a domain node
 * - check that an object of a subclass can be used as one of the superclass
 * - test map_descendant_bottom_up in the successful case
 */
static isl::schedule_node test_schedule_tree_generic(isl::ctx ctx)
{
	auto schedule = construct_schedule_tree(ctx);
	auto root = schedule.root();

	assert(IS_TRUE(root.isa<isl::schedule_node_domain>()));
	root = root.as<isl::schedule_node_domain>().child(0).parent();

	int count = 0;
	auto inc_count = [&count](isl::schedule_node node) {
		count++;
		return node;
	};
	root = root.map_descendant_bottom_up(inc_count);
	assert(count == 8);

	return root;
}

/* Test marking band members for unrolling.
 * "schedule" is the schedule created by construct_schedule_tree.
 * It schedules two statements, with 10 and 20 instances, respectively.
 * Unrolling all band members therefore results in 30 at-domain calls
 * by the AST generator.
 */
static void test_ast_build_unroll(isl::schedule schedule)
{
	auto root = schedule.root();
	auto mark_unroll = [](isl::schedule_node node) {
		if (IS_TRUE(node.isa<isl::schedule_node_band>())) {
			auto band = node.as<isl::schedule_node_band>();
			node = band.member_set_ast_loop_unroll(0);
		}
		return node;
	};
	root = root.map_descendant_bottom_up(mark_unroll);
	schedule = root.schedule();

	int count_ast = 0;
	auto inc_count_ast =
	    [&count_ast](isl::ast_node node, isl::ast_build build) {
		count_ast++;
		return node;
	};
	auto build = isl::ast_build(schedule.ctx());
	build = build.set_at_each_domain(inc_count_ast);
	auto ast = build.node_from(schedule);
	assert(count_ast == 30);
}

/* Test basic AST generation from a schedule tree that is independent
 * of the type of bindings.
 *
 * In particular, create a simple schedule tree and
 * - generate an AST from the schedule tree
 * - test at_each_domain in the successful case
 * - test unrolling
 */
static isl::schedule test_ast_build_generic(isl::ctx ctx)
{
	auto schedule = construct_schedule_tree(ctx);

	int count_ast = 0;
	auto inc_count_ast =
	    [&count_ast](isl::ast_node node, isl::ast_build build) {
		count_ast++;
		return node;
	};
	auto build = isl::ast_build(ctx);
	auto build_copy = build.set_at_each_domain(inc_count_ast);
	auto ast = build.node_from(schedule);
	assert(count_ast == 0);
	count_ast = 0;
	ast = build_copy.node_from(schedule);
	assert(count_ast == 2);
	build = build_copy;
	count_ast = 0;
	ast = build.node_from(schedule);
	assert(count_ast == 2);

	test_ast_build_unroll(schedule);

	return schedule;
}

/* Test basic AST expression generation from an affine expression.
 */
static void test_ast_build_expr(isl::ctx ctx)
{
	isl::pw_aff pa(ctx, "[n] -> { [n + 1] }");
	isl::ast_build build = isl::ast_build::from_context(pa.domain());

	auto expr = build.expr_from(pa);
	auto op = expr.as<isl::ast_expr_op>();
	assert(IS_TRUE(op.isa<isl::ast_expr_op_add>()));
	assert(SIZE_VAL(op.n_arg()) == 2);
}
