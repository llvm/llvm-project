/*
 * Copyright 2008-2009 Katholieke Universiteit Leuven
 * Copyright 2010      INRIA Saclay
 * Copyright 2012-2013 Ecole Normale Superieure
 * Copyright 2014      INRIA Rocquencourt
 * Copyright 2021-2022 Cerebras Systems
 *
 * Use of this software is governed by the MIT license
 *
 * Written by Sven Verdoolaege, K.U.Leuven, Departement
 * Computerwetenschappen, Celestijnenlaan 200A, B-3001 Leuven, Belgium
 * and INRIA Saclay - Ile-de-France, Parc Club Orsay Universite,
 * ZAC des vignes, 4 rue Jacques Monod, 91893 Orsay, France
 * and Ecole Normale Superieure, 45 rue d'Ulm, 75230 Paris, France
 * and Inria Paris - Rocquencourt, Domaine de Voluceau - Rocquencourt,
 * B.P. 105 - 78153 Le Chesnay, France
 * and Cerebras Systems, 1237 E Arques Ave, Sunnyvale, CA, USA
 */

#include <assert.h>
#include <stdlib.h>

#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <isl/cpp.h>

/* A binary isl function that appears in the C++ bindings
 * as a unary method in a class T, taking an extra argument
 * of type A1 and returning an object of type R.
 */
template <typename A1, typename R, typename T>
using binary_fn = R (T::*)(A1) const;

/* A function for selecting an overload of a pointer to a unary C++ method
 * based on the single argument type.
 * The object type and the return type are meant to be deduced.
 */
template <typename A1, typename R, typename T>
static binary_fn<A1, R, T> const arg(const binary_fn<A1, R, T> &fn)
{
	return fn;
}

/* A ternary isl function that appears in the C++ bindings
 * as a binary method in a class T, taking extra arguments
 * of type A1 and A2 and returning an object of type R.
 */
template <typename A1, typename A2, typename R, typename T>
using ternary_fn = R (T::*)(A1, A2) const;

/* A function for selecting an overload of a pointer to a binary C++ method
 * based on the (first) argument type(s).
 * The object type and the return type are meant to be deduced.
 */
template <typename A1, typename A2, typename R, typename T>
static ternary_fn<A1, A2, R, T> const arg(const ternary_fn<A1, A2, R, T> &fn)
{
	return fn;
}

/* A description of the input and the output of a unary operation.
 */
struct unary {
	const char *arg;
	const char *res;
};

/* A description of the inputs and the output of a binary operation.
 */
struct binary {
	const char *arg1;
	const char *arg2;
	const char *res;
};

/* A description of the inputs and the output of a ternary operation.
 */
struct ternary {
	const char *arg1;
	const char *arg2;
	const char *arg3;
	const char *res;
};

/* A template function for checking whether two objects
 * of the same (isl) type are (obviously) equal.
 * The spelling depends on the isl type and
 * in particular on whether an equality method is available or
 * whether only obvious equality can be tested.
 */
template <typename T, typename std::decay<decltype(
	std::declval<T>().is_equal(std::declval<T>()))>::type = true>
static bool is_equal(const T &a, const T &b)
{
	return a.is_equal(b);
}
template <typename T, typename std::decay<decltype(
	std::declval<T>().plain_is_equal(std::declval<T>()))>::type = true>
static bool is_equal(const T &a, const T &b)
{
	return a.plain_is_equal(b);
}

/* A helper macro for throwing an isl::exception_invalid with message "msg".
 */
#define THROW_INVALID(msg) \
	isl::exception::throw_error(isl_error_invalid, msg, __FILE__, __LINE__)

/* Run a sequence of tests of method "fn" with stringification "name" and
 * with input and output described by "test",
 * throwing an exception when an unexpected result is produced.
 */
template <typename R, typename T>
static void test(isl::ctx ctx, R (T::*fn)() const, const std::string &name,
	const std::vector<unary> &tests)
{
	for (const auto &test : tests) {
		T obj(ctx, test.arg);
		R expected(ctx, test.res);
		const auto &res = (obj.*fn)();
		std::ostringstream ss;

		if (is_equal(expected, res))
			continue;

		ss << name << "(" << test.arg << ") =\n"
		   << res << "\n"
		   << "expecting:\n"
		   << expected;
		THROW_INVALID(ss.str().c_str());
	}
}

/* Run a sequence of tests of method "fn" with stringification "name" and
 * with inputs and output described by "test",
 * throwing an exception when an unexpected result is produced.
 */
template <typename R, typename T, typename A1>
static void test(isl::ctx ctx, R (T::*fn)(A1) const, const std::string &name,
	const std::vector<binary> &tests)
{
	for (const auto &test : tests) {
		T obj(ctx, test.arg1);
		A1 arg1(ctx, test.arg2);
		R expected(ctx, test.res);
		const auto &res = (obj.*fn)(arg1);
		std::ostringstream ss;

		if (is_equal(expected, res))
			continue;

		ss << name << "(" << test.arg1 << ", " << test.arg2 << ") =\n"
		   << res << "\n"
		   << "expecting:\n"
		   << test.res;
		THROW_INVALID(ss.str().c_str());
	}
}

/* Run a sequence of tests of method "fn" with stringification "name" and
 * with inputs and output described by "test",
 * throwing an exception when an unexpected result is produced.
 */
template <typename R, typename T, typename A1, typename A2>
static void test(isl::ctx ctx, R (T::*fn)(A1, A2) const,
	const std::string &name, const std::vector<ternary> &tests)
{
	for (const auto &test : tests) {
		T obj(ctx, test.arg1);
		A1 arg1(ctx, test.arg2);
		A2 arg2(ctx, test.arg3);
		R expected(ctx, test.res);
		const auto &res = (obj.*fn)(arg1, arg2);
		std::ostringstream ss;

		if (is_equal(expected, res))
			continue;

		ss << name << "(" << test.arg1 << ", " << test.arg2 << ", "
		   << test.arg3 << ") =\n"
		   << res << "\n"
		   << "expecting:\n"
		   << expected;
		THROW_INVALID(ss.str().c_str());
	}
}

/* A helper macro that calls test with as implicit initial argument "ctx" and
 * as extra argument a stringification of "FN".
 */
#define C(FN, ...) test(ctx, FN, #FN, __VA_ARGS__)

/* Perform some basic isl::space tests.
 */
static void test_space(isl::ctx ctx)
{
	C(&isl::space::domain, {
	{ "{ A[] -> B[] }", "{ A[] }" },
	{ "{ A[C[] -> D[]] -> B[E[] -> F[]] }", "{ A[C[] -> D[]] }" },
	});

	C(&isl::space::range, {
	{ "{ A[] -> B[] }", "{ B[] }" },
	{ "{ A[C[] -> D[]] -> B[E[] -> F[]] }", "{ B[E[] -> F[]] }" },
	});

	C(&isl::space::params, {
	{ "{ A[] -> B[] }", "{ : }" },
	{ "{ A[C[] -> D[]] -> B[E[] -> F[]] }", "{ : }" },
	});
}

/* Perform some basic conversion tests.
 */
static void test_conversion(isl::ctx ctx)
{
	C(&isl::multi_pw_aff::as_set, {
	{ "[n] -> { [] : n >= 0 } ",
	  "[n] -> { [] : n >= 0 } " },
	});
}

/* Perform some basic preimage tests.
 */
static void test_preimage(isl::ctx ctx)
{
	C(arg<isl::multi_aff>(&isl::set::preimage), {
	{ "{ B[i,j] : 0 <= i < 10 and 0 <= j < 100 }",
	  "{ A[j,i] -> B[i,j] }",
	  "{ A[j,i] : 0 <= i < 10 and 0 <= j < 100 }" },
	{ "{ rat: B[i,j] : 0 <= i, j and 3 i + 5 j <= 100 }",
	  "{ A[a,b] -> B[a/2,b/6] }",
	  "{ rat: A[a,b] : 0 <= a, b and 9 a + 5 b <= 600 }" },
	{ "{ B[i,j] : 0 <= i, j and 3 i + 5 j <= 100 }",
	  "{ A[a,b] -> B[a/2,b/6] }",
	  "{ A[a,b] : 0 <= a, b and 9 a + 5 b <= 600 and "
		    "exists i,j : a = 2 i and b = 6 j }" },
	{ "[n] -> { S[i] : 0 <= i <= 100 }", "[n] -> { S[n] }",
	  "[n] -> { : 0 <= n <= 100 }" },
	{ "{ B[i] : 0 <= i < 100 and exists a : i = 4 a }",
	  "{ A[a] -> B[2a] }",
	  "{ A[a] : 0 <= a < 50 and exists b : a = 2 b }" },
	{ "{ B[i] : 0 <= i < 100 and exists a : i = 4 a }",
	  "{ A[a] -> B[([a/2])] }",
	  "{ A[a] : 0 <= a < 200 and exists b : [a/2] = 4 b }" },
	{ "{ B[i,j,k] : 0 <= i,j,k <= 100 }",
	  "{ A[a] -> B[a,a,a/3] }",
	  "{ A[a] : 0 <= a <= 100 and exists b : a = 3 b }" },
	{ "{ B[i,j] : j = [(i)/2] } ", "{ A[i,j] -> B[i/3,j] }",
	  "{ A[i,j] : j = [(i)/6] and exists a : i = 3 a }" },
	});

	C(arg<isl::multi_aff>(&isl::union_map::preimage_domain), {
	{ "{ B[i,j] -> C[2i + 3j] : 0 <= i < 10 and 0 <= j < 100 }",
	  "{ A[j,i] -> B[i,j] }",
	  "{ A[j,i] -> C[2i + 3j] : 0 <= i < 10 and 0 <= j < 100 }" },
	{ "{ B[i] -> C[i]; D[i] -> E[i] }",
	  "{ A[i] -> B[i + 1] }",
	  "{ A[i] -> C[i + 1] }" },
	{ "{ B[i] -> C[i]; B[i] -> E[i] }",
	  "{ A[i] -> B[i + 1] }",
	  "{ A[i] -> C[i + 1]; A[i] -> E[i + 1] }" },
	{ "{ B[i] -> C[([i/2])] }",
	  "{ A[i] -> B[2i] }",
	  "{ A[i] -> C[i] }" },
	{ "{ B[i,j] -> C[([i/2]), ([(i+j)/3])] }",
	  "{ A[i] -> B[([i/5]), ([i/7])] }",
	  "{ A[i] -> C[([([i/5])/2]), ([(([i/5])+([i/7]))/3])] }" },
	{ "[N] -> { B[i] -> C[([N/2]), i, ([N/3])] }",
	  "[N] -> { A[] -> B[([N/5])] }",
	  "[N] -> { A[] -> C[([N/2]), ([N/5]), ([N/3])] }" },
	{ "{ B[i] -> C[i] : exists a : i = 5 a }",
	  "{ A[i] -> B[2i] }",
	  "{ A[i] -> C[2i] : exists a : 2i = 5 a }" },
	{ "{ B[i] -> C[i] : exists a : i = 2 a; "
	    "B[i] -> D[i] : exists a : i = 2 a + 1 }",
	  "{ A[i] -> B[2i] }",
	  "{ A[i] -> C[2i] }" },
	{ "{ A[i] -> B[i] }", "{ C[i] -> A[(i + floor(i/3))/2] }",
	  "{ C[i] -> B[j] : 2j = i + floor(i/3) }" },
	});

	C(arg<isl::multi_aff>(&isl::union_map::preimage_range), {
	{ "[M] -> { A[a] -> B[a] }", "[M] -> { C[] -> B[floor(M/2)] }",
	  "[M] -> { A[floor(M/2)] -> C[] }" },
	});
}

/* Perform some basic fixed power tests.
 */
static void test_fixed_power(isl::ctx ctx)
{
	C(arg<isl::val>(&isl::map::fixed_power), {
	{ "{ [i] -> [i + 1] }", "23",
	  "{ [i] -> [i + 23] }" },
	{ "{ [a = 0:1, b = 0:15, c = 0:1, d = 0:1, 0] -> [a, b, c, d, 1]; "
	    "[a = 0:1, b = 0:15, c = 0:1, 0, 1] -> [a, b, c, 1, 0];  "
	    "[a = 0:1, b = 0:15, 0, 1, 1] -> [a, b, 1, 0, 0];  "
	    "[a = 0:1, b = 0:14, 1, 1, 1] -> [a, 1 + b, 0, 0, 0];  "
	    "[0, 15, 1, 1, 1] -> [1, 0, 0, 0, 0] }",
	  "128",
	  "{ [0, b = 0:15, c = 0:1, d = 0:1, e = 0:1] -> [1, b, c, d, e] }" },
	});
}

/* Perform some basic intersection tests.
 */
static void test_intersect(isl::ctx ctx)
{
	C(&isl::union_map::intersect_domain_wrapped_domain, {
	{ "{ [A[x] -> B[y]] -> C[z]; [D[x] -> A[y]] -> E[z] }",
	  "{ A[0] }",
	  "{ [A[0] -> B[y]] -> C[z] }" },
	{ "{ C[z] -> [A[x] -> B[y]]; E[z] -> [D[x] -> A[y]] }",
	  "{ A[0] }",
	  "{ }" },
	});

	C(&isl::union_map::intersect_range_wrapped_domain, {
	{ "{ [A[x] -> B[y]] -> C[z]; [D[x] -> A[y]] -> E[z] }",
	  "{ A[0] }",
	  "{ }" },
	{ "{ C[z] -> [A[x] -> B[y]]; E[z] -> [D[x] -> A[y]] }",
	  "{ A[0] }",
	  "{ C[z] -> [A[0] -> B[y]] }" },
	});
}

/* Perform some basic gist tests.
 */
static void test_gist(isl::ctx ctx)
{
	C(&isl::pw_aff::gist_params, {
	{ "[N] -> { D[x] -> [x] : N >= 0; D[x] -> [0] : N < 0 }",
	  "[N] -> { : N >= 0 }",
	  "[N] -> { D[x] -> [x] }" },
	});
}

/* Perform tests that project out parameters.
 */
static void test_project(isl::ctx ctx)
{
	C(arg<isl::id>(&isl::union_map::project_out_param), {
	{ "[N] -> { D[i] -> A[0:N-1]; D[i] -> B[i] }", "N",
	  "{ D[i] -> A[0:]; D[i] -> B[i] }" },
	{ "[N] -> { D[i] -> A[0:N-1]; D[i] -> B[i] }", "M",
	  "[N] -> { D[i] -> A[0:N-1]; D[i] -> B[i] }" },
	});

	C(arg<isl::id_list>(&isl::union_map::project_out_param), {
	{ "[M, N, O] -> { D[i] -> A[j] : i <= j < M, N, O }", "(M, N)",
	  "[O] -> { D[i] -> A[j] : i <= j < O }" },
	});
}

/* Perform some basic scaling tests.
 */
static void test_scale(isl::ctx ctx)
{
	C(arg<isl::multi_val>(&isl::pw_multi_aff::scale), {
	{ "{ A[a] -> B[a, a + 1, a - 1] : a >= 0 }", "{ B[2, 7, 0] }",
	  "{ A[a] -> B[2a, 7a + 7, 0] : a >= 0 }" },
	});
	C(arg<isl::multi_val>(&isl::pw_multi_aff::scale), {
	{ "{ A[a] -> B[1, a - 1] : a >= 0 }", "{ B[1/2, 7] }",
	  "{ A[a] -> B[1/2, 7a - 7] : a >= 0 }" },
	});

	C(arg<isl::multi_val>(&isl::pw_multi_aff::scale_down), {
	{ "{ A[a] -> B[a, a + 1] : a >= 0 }", "{ B[2, 7] }",
	  "{ A[a] -> B[a/2, (a + 1)/7] : a >= 0 }" },
	});
	C(arg<isl::multi_val>(&isl::pw_multi_aff::scale_down), {
	{ "{ A[a] -> B[a, a - 1] : a >= 0 }", "{ B[2, 1/7] }",
	  "{ A[a] -> B[a/2, 7a - 7] : a >= 0 }" },
	});
}

/* Perform some basic isl::id_to_id tests.
 */
static void test_id_to_id(isl::ctx ctx)
{
	C((arg<isl::id, isl::id>(&isl::id_to_id::set)), {
	{ "{ }", "a", "b",
	  "{ a: b }" },
	{ "{ a: b }", "a", "b",
	  "{ a: b }" },
	{ "{ a: c }", "a", "b",
	  "{ a: b }" },
	{ "{ a: b }", "b", "a",
	  "{ a: b, b: a }" },
	{ "{ a: b }", "b", "a",
	  "{ b: a, a: b }" },
	});
}

/* The list of tests to perform.
 */
static std::vector<std::pair<const char *, void (*)(isl::ctx)>> tests =
{
	{ "space", &test_space },
	{ "conversion", &test_conversion },
	{ "preimage", &test_preimage },
	{ "fixed power", &test_fixed_power },
	{ "intersect", &test_intersect },
	{ "gist", &test_gist },
	{ "project out parameters", &test_project },
	{ "scale", &test_scale },
	{ "id-to-id", &test_id_to_id },
};

/* Perform some basic checks by means of the C++ bindings.
 */
int main(int argc, char **argv)
{
	int ret = EXIT_SUCCESS;
	struct isl_ctx *ctx;
	struct isl_options *options;

	options = isl_options_new_with_defaults();
	assert(options);
	argc = isl_options_parse(options, argc, argv, ISL_ARG_ALL);
	ctx = isl_ctx_alloc_with_options(&isl_options_args, options);

	try {
		for (const auto &f : tests) {
			std::cout << f.first << "\n";
			f.second(ctx);
		}
	} catch (const isl::exception &e) {
		std::cerr << e.what() << "\n";
		ret = EXIT_FAILURE;
	}

	isl_ctx_free(ctx);
	return ret;
}
