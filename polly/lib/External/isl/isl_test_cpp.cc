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

#include <map>

#include <isl/options.h>
#include <isl/typed_cpp.h>

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
#undef assert
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

/* Test the functionality of "foreach_scc" functions.
 *
 * In particular, test it on a list of elements that can be completely sorted
 * but where two of the elements ("a" and "b") are incomparable.
 */
static void test_foreach_scc(isl::ctx ctx)
{
	isl::multi_pw_aff id;
	isl::id_list list(ctx, 3);
	isl::id_list sorted(ctx, 3);
	std::map<std::string, isl::map> data = {
		{ "a", isl::map(ctx, "{ [0] -> [1] }") },
		{ "b", isl::map(ctx, "{ [1] -> [0] }") },
		{ "c", isl::map(ctx, "{ [i = 0:1] -> [i] }") },
	};

	for (const auto &kvp: data)
		list = list.add(kvp.first);
	id = data.at("a").space().domain().identity_multi_pw_aff_on_domain();
	list.foreach_scc([&data, &id] (isl::id a, isl::id b) {
		auto map = data.at(b.name()).apply_domain(data.at(a.name()));
		return !map.lex_ge_at(id).is_empty();
	}, [&sorted] (isl::id_list scc) {
		assert(scc.size() == 1);
		sorted = sorted.concat(scc);
	});
	assert(sorted.size() == 3);
	assert(sorted.at(0).name() == "b");
	assert(sorted.at(1).name() == "c");
	assert(sorted.at(2).name() == "a");
}

/* Test the functionality of "every" functions.
 *
 * In particular, test the generic functionality and
 * test that exceptions are properly propagated.
 */
static void test_every(isl::ctx ctx)
{
	isl::union_set us(ctx, "{ A[i]; B[j] }");

	test_every_generic(ctx);

	auto fail = [] (isl::set s) -> bool {
		throw "fail";
	};
	bool caught = false;
	try {
		us.every_set(fail);
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

/* Test basic schedule tree functionality.
 *
 * In particular, create a simple schedule tree and
 * - perform some generic tests
 * - test map_descendant_bottom_up in the failing case
 * - test foreach_descendant_top_down
 * - test every_descendant
 */
static void test_schedule_tree(isl::ctx ctx)
{
	auto root = test_schedule_tree_generic(ctx);

	auto fail_map = [](isl::schedule_node node) {
		throw "fail";
		return node;
	};
	auto caught = false;
	try {
		root.map_descendant_bottom_up(fail_map);
		die("no exception raised");
	} catch (char const *s) {
		caught = true;
	}
	assert(caught);

	int count = 0;
	auto inc_count = [&count](isl::schedule_node node) {
		count++;
		return true;
	};
	root.foreach_descendant_top_down(inc_count);
	assert(count == 8);

	count = 0;
	auto inc_count_once = [&count](isl::schedule_node node) {
		count++;
		return false;
	};
	root.foreach_descendant_top_down(inc_count_once);
	assert(count == 1);

	auto is_not_domain = [](isl::schedule_node node) {
		return !node.isa<isl::schedule_node_domain>();
	};
	assert(root.child(0).every_descendant(is_not_domain));
	assert(!root.every_descendant(is_not_domain));

	auto fail = [](isl::schedule_node node) {
		throw "fail";
		return true;
	};
	caught = false;
	try {
		root.every_descendant(fail);
		die("no exception raised");
	} catch (char const *s) {
		caught = true;
	}
	assert(caught);

	auto domain = root.as<isl::schedule_node_domain>().domain();
	auto filters = isl::union_set(ctx, "{}");
	auto collect_filters = [&filters](isl::schedule_node node) {
		if (node.isa<isl::schedule_node_filter>()) {
			auto filter = node.as<isl::schedule_node_filter>();
			filters = filters.unite(filter.filter());
		}
		return true;
	};
	root.every_descendant(collect_filters);
	assert(domain.is_equal(filters));
}

/* Test basic AST generation from a schedule tree.
 *
 * In particular, create a simple schedule tree and
 * - perform some generic tests
 * - test at_each_domain in the failing case
 */
static void test_ast_build(isl::ctx ctx)
{
	auto schedule = test_ast_build_generic(ctx);

	bool do_fail = true;
	int count_ast_fail = 0;
	auto fail_inc_count_ast =
	    [&count_ast_fail, &do_fail](isl::ast_node node,
					isl::ast_build build) {
		count_ast_fail++;
		if (do_fail)
			throw "fail";
		return node;
	};
	auto build = isl::ast_build(ctx);
	build = build.set_at_each_domain(fail_inc_count_ast);
	auto caught = false;
	try {
		auto ast = build.node_from(schedule);
	} catch (char const *s) {
		caught = true;
	}
	assert(caught);
	assert(count_ast_fail > 0);
	auto build_copy = build;
	int count_ast = 0;
	auto inc_count_ast =
	    [&count_ast](isl::ast_node node, isl::ast_build build) {
		count_ast++;
		return node;
	};
	build_copy = build_copy.set_at_each_domain(inc_count_ast);
	auto ast = build_copy.node_from(schedule);
	assert(count_ast == 2);
	count_ast_fail = 0;
	do_fail = false;
	ast = build.node_from(schedule);
	assert(count_ast_fail == 2);
}

/* Basic test of the templated interface.
 *
 * Intersecting the domain of an access relation
 * with statement instances should be allowed,
 * while intersecting the range with statement instances
 * should result in a compile-time error.
 */
static void test_typed(isl::ctx ctx)
{
	struct ST {};
	struct AR {};
	isl::typed::map<ST, AR> access(ctx, "{ S[i, j] -> A[i] }");
	isl::typed::set<ST> instances(ctx, "{ S[i, j] : 0 <= i, j < 10 }");

#ifndef COMPILE_ERROR
	access.intersect_domain(instances);
#else
	access.intersect_range(instances);
#endif
}

/* Test the (unchecked) isl C++ interface
 *
 * This includes:
 *  - The isl C <-> C++ pointer interface
 *  - Object construction
 *  - Different parameter types
 *  - Different return types
 *  - Foreach functions
 *  - Foreach SCC function
 *  - Exceptions
 *  - Spaces
 *  - Schedule trees
 *  - AST generation
 *  - AST expression generation
 *  - Templated interface
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
	test_foreach_scc(ctx);
	test_every(ctx);
	test_exception(ctx);
	test_space(ctx);
	test_schedule_tree(ctx);
	test_ast_build(ctx);
	test_ast_build_expr(ctx);
	test_typed(ctx);

	isl_ctx_free(ctx);

	return EXIT_SUCCESS;
}
