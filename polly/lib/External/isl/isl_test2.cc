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
#include <ios>
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

/* A description of the input and the output of a unary property.
 */
struct unary_prop {
	const char *arg;
	bool res;
};

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

/* Run a sequence of tests of function "fn" with stringification "name" and
 * with input and output described by "tests",
 * throwing an exception when an unexpected result is produced.
 */
template <typename T>
static void test(isl::ctx ctx, bool fn(const T &), const std::string &name,
	const std::vector<unary_prop> &tests)
{
	for (const auto &test : tests) {
		T obj(ctx, test.arg);
		bool res = fn(obj);
		std::ostringstream ss;

		if (test.res == res)
			continue;

		ss << name << "(" << test.arg << ") = "
		   << std::boolalpha << res << "\n"
		   << "expecting: "
		   << test.res;
		THROW_INVALID(ss.str().c_str());
	}
}

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
		   << expected;
		THROW_INVALID(ss.str().c_str());
	}
}

/* Run a sequence of tests of function "fn" with stringification "name" and
 * with inputs and output described by "tests",
 * throwing an exception when an unexpected result is produced.
 */
template <typename R, typename T, typename A1, typename A2, typename F>
static void test_ternary(isl::ctx ctx, const F &fn,
	const std::string &name, const std::vector<ternary> &tests)
{
	for (const auto &test : tests) {
		T obj(ctx, test.arg1);
		A1 arg1(ctx, test.arg2);
		A2 arg2(ctx, test.arg3);
		R expected(ctx, test.res);
		const auto &res = fn(obj, arg1, arg2);
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

/* Run a sequence of tests of function "fn" with stringification "name" and
 * with inputs and output described by "tests",
 * throwing an exception when an unexpected result is produced.
 *
 * Simply call test_ternary.
 */
template <typename R, typename T, typename A1, typename A2>
static void test(isl::ctx ctx, R fn(const T&, const A1&, const A2&),
	const std::string &name, const std::vector<ternary> &tests)
{
	test_ternary<R, T, A1, A2>(ctx, fn, name, tests);
}

/* Run a sequence of tests of method "fn" with stringification "name" and
 * with inputs and output described by "tests",
 * throwing an exception when an unexpected result is produced.
 *
 * Wrap the method pointer into a function taking an object reference and
 * call test_ternary.
 */
template <typename R, typename T, typename A1, typename A2>
static void test(isl::ctx ctx, R (T::*fn)(A1, A2) const,
	const std::string &name, const std::vector<ternary> &tests)
{
	const auto &wrap = [&] (const T &o, const A1 &arg1, const A2 &arg2) {
		return (o.*fn)(arg1, arg2);
	};
	test_ternary<R, T, A1, A2>(ctx, wrap, name, tests);
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

/* Is "fn" an expression defined over a single cell?
 */
static bool has_single_cell(const isl::pw_multi_aff &fn)
{
	const auto &domain = fn.domain();
	return fn.gist(domain).isa_multi_aff();
}

/* Does the conversion of "obj" to an isl_pw_multi_aff
 * result in an expression defined over a single cell?
 */
template <typename T>
static bool has_single_cell_pma(const T &obj)
{
	return has_single_cell(obj.as_pw_multi_aff());
}

/* Perform some basic conversion tests.
 *
 * In particular, check that a map with an output dimension
 * that is equal to some integer division over a domain involving
 * a local variable without a known integer division expression or
 * to some linear combination of integer divisions
 * can be converted to a function expressed in the same way.
 *
 * Also, check that a nested modulo expression can be extracted
 * from a set or binary relation representation, or at least
 * that a conversion to a function does not result in multiple cells.
 */
static void test_conversion(isl::ctx ctx)
{
	C(&isl::set::as_pw_multi_aff, {
	{ "[N=0:] -> { [] }",
	  "[N=0:] -> { [] }" },
	});

	C(&isl::multi_pw_aff::as_set, {
	{ "[n] -> { [] : n >= 0 } ",
	  "[n] -> { [] : n >= 0 } " },
	});

	C(&isl::map::as_pw_multi_aff, {
	{ "{ [a] -> [a//2] : "
	    "exists (e0: 8*floor((-a + e0)/8) <= -8 - a + 8e0) }",
	  "{ [a] -> [a//2] : "
	    "exists (e0: 8*floor((-a + e0)/8) <= -8 - a + 8e0) }" },
	{ "{ [a, b] -> [(2*floor((a)/8) + floor((b)/6))] }",
	  "{ [a, b] -> [(2*floor((a)/8) + floor((b)/6))] }" },
	});

	C(&has_single_cell_pma<isl::set>, {
	{ "[s=0:23] -> { A[(s//4)%3, s%4, s//12] }", true },
	});

	C(&has_single_cell_pma<isl::map>, {
	{ "{ [a] -> [a//2] : "
	    "exists (e0: 8*floor((-a + e0)/8) <= -8 - a + 8e0) }",
	  true },
	{ "{ [s=0:23, t] -> B[((s+1+2t)//4)%3, 2+(s+1+2t)%4, (s+1+2t)//12] }",
	  true },
	{ "{ [a=0:31] -> [b=0:3, c] : 4c = 28 - a + b }", true },
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

	C(arg<isl::pw_multi_aff>(&isl::set::preimage), {
	{ "{ B[i,j] : 0 <= i < 10 and 0 <= j < 100 }",
	  "{ A[j,i] -> B[i,j] : false }",
	  "{ A[j,i] : false }" },
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
	C(arg<isl::basic_set>(&isl::basic_map::intersect_params), {
	{ "[n] -> { A[x] -> B[y] }", "[n] -> { : n >= 0 }",
	  "[n] -> { A[x] -> B[y] : n >= 0 }" },
	});

	C(&isl::union_map::intersect_domain_wrapped_domain, {
	{ "{ [A[x] -> B[y]] -> C[z]; [D[x] -> A[y]] -> E[z] }",
	  "{ A[0] }",
	  "{ [A[0] -> B[y]] -> C[z] }" },
	{ "{ C[z] -> [A[x] -> B[y]]; E[z] -> [D[x] -> A[y]] }",
	  "{ A[0] }",
	  "{ }" },
	{ "{ T[A[x] -> B[y]] -> C[z]; [D[x] -> A[y]] -> E[z] }",
	  "{ A[0] }",
	  "{ T[A[0] -> B[y]] -> C[z] }" },
	});

	C(&isl::union_map::intersect_range_wrapped_domain, {
	{ "{ [A[x] -> B[y]] -> C[z]; [D[x] -> A[y]] -> E[z] }",
	  "{ A[0] }",
	  "{ }" },
	{ "{ C[z] -> [A[x] -> B[y]]; E[z] -> [D[x] -> A[y]] }",
	  "{ A[0] }",
	  "{ C[z] -> [A[0] -> B[y]] }" },
	{ "{ C[z] -> T[A[x] -> B[y]]; E[z] -> [D[x] -> A[y]] }",
	  "{ A[0] }",
	  "{ C[z] -> T[A[0] -> B[y]] }" },
	});
}

/* Is the expression for the lexicographic minimum of "obj"
 * defined over a single cell?
 */
template <typename T>
static bool lexmin_has_single_cell(const T &obj)
{
	return has_single_cell(obj.lexmin_pw_multi_aff());
}

/* Perform some basic lexicographic minimization tests.
 */
static void test_lexmin(isl::ctx ctx)
{
	C(&lexmin_has_single_cell<isl::map>, {
	/* The following two inputs represent the same binary relation,
	 * the second with extra redundant constraints.
	 * The lexicographic minimum of both should consist of a single cell.
	 */
	{ "{ [a=0:11] -> [b] : -1 + b <= 2*floor((a)/6) <= b }", true },
	{ "{ [a=0:11] -> [b=0:3] : -1 + b <= 2*floor((a)/6) <= b }", true },

	{ "{ [a = 0:2, b = 0:1] -> [c = 0:9, d = (-a + b) mod 3] : "
	    "10a + 5b - 3c <= 5d <= 12 + 10a + 5b - 3c }", true },
	{ "{ [a=0:71] -> [(a//3)%8] }", true },
	{ "{ [a=0:71] -> [b=0:7] : (a - 3 * b + 21) % 24 >= 21 }", true },
	{ "{ [a=0:71] -> [b=0:7] : (a - 3 * b + 21) % 24 >= 20 }", false },
	{ "{ [a=0:71] -> [b=0:7] : (a - 3 * b + 21) % 24 >= 22 }", true },
	{ "{ [a=0:71] -> [b=-7:0] : (a + 3 * b + 21) % 24 >= 21 }", true },
	{ "{ [a=0:71] -> [b=-7:0] : (a + 3 * b + 21) % 24 >= 20 }", false },
	{ "{ [a=0:71] -> [b=-7:0] : (a + 3 * b + 21) % 24 >= 22 }", true },
	});

	C(&isl::map::lexmin_pw_multi_aff, {
	/* The following two inputs represent the same binary relation,
	 * the second with some redundant constraints removed.
	 * The lexicographic minimum of both should consist of a single cell.
	 */
	{ "{ [a=0:3] -> [b=a//2] : 0 <= b <= 1 }",
	  "{ [a=0:3] -> [(floor((a)/2))] }" },
	{ "{ [a] -> [b=a//2] : 0 <= b <= 1 }",
	  "{ [a=0:3] -> [(floor((a)/2))] }" },

	{ "{ [a = 0:2, b = 0:1] -> [c = 0:9, d = (-a + b) mod 3] : "
	    "10a + 5b - 3c <= 5d <= 12 + 10a + 5b - 3c }",
	  "{ [a = 0:2, b = 0:1] -> [5*(2a + b)//3, (2a + b) mod 3] }" },
	{ "{ [a=0:71] -> [(a//3)%8] }",
	  "{ [a=0:71] -> [(a//3)%8] }" },
	{ "{ [a=0:71] -> [b=0:7] : (a - 3 * b + 21) % 24 >= 21 }",
	  "{ [a=0:71] -> [(a//3)%8] }" },
	{ "{ [a=0:71] -> [b=0:7] : (a - 3 * b + 21) % 24 >= 22 }",
	  "{ [a=0:71] -> [(a//3)%8] : a % 3 > 0 }" },
	{ "{ [a=0:71] -> [b=-7:0] : (a + 3 * b + 21) % 24 >= 21 }",
	  "{ [a=0:71] -> [(-7 + (-1 - floor((a)/3)) mod 8)] }" },
	});

	C(&isl::set::lexmin_pw_multi_aff, {
	{ "[a] -> { [b=a//2] : 0 <= b <= 1 }",
	  "[a=0:3] -> { [(floor((a)/2))] }" },
	{ "[a=0:71] -> { [(a//3)%8] }",
	  "[a=0:71] -> { [(a//3)%8] }" },
	{ "[a=0:71] -> { [b=0:7] : (a - 3 * b + 21) % 24 >= 21 }",
	  "[a=0:71] -> { [(a//3)%8] }" },
	});
}

/* Compute the gist of "obj" with respect to "context",
 * with "copy" an independent copy of "obj",
 * but also check that applying the gist operation does
 * not modify the input set (an earlier version of isl would do that) and
 * that the test case is consistent, i.e., that the gist has the same
 * intersection with the context as the input set.
 */
template <typename T>
T gist(const T &obj, const T &copy, const T &context)
{
	const auto &res = obj.gist(context);
	if (!is_equal(obj, copy)) {
		std::ostringstream ss;
		ss << "gist changed " << copy << " into " << obj;
		THROW_INVALID(ss.str().c_str());
	}
	if (!is_equal(obj.intersect(context), res.intersect(context))) {
		std::ostringstream ss;
		ss << "inconsistent "
		   << obj << " % " << context << " = " << res;
		THROW_INVALID(ss.str().c_str());
	}
	return res;
}

/* A helper macro for producing two instances of "x".
 */
#define TWO(x)	(x), (x)

/* Perform some basic gist tests.
 *
 * The gist() function is given two identical inputs so that
 * it can check that the input to the call to the gist method
 * is not modified.
 */
static void test_gist(isl::ctx ctx)
{
	C(&gist<isl::basic_set> , {
	{ TWO("{ [i=100:] }"),
	  "{ [i] : exists a, b: 2b > 2i - 5a > 8b -3 i and 3b > 2a }",
	  "{ [i=100:] }" },
	{ TWO("{ [i=0:] }"),
	  "{ [i] : exists a, b: 2b > 2i - 5a > 8b -3 i and 3b > 2a }",
	  "{ [i] }" },
	{ TWO("{ [i] : exists (e0, e1: 3e1 >= 1 + 2e0 and "
	    "8e1 <= -1 + 5i - 5e0 and 2e1 >= 1 + 2i - 5e0) }"),
	  "{ [i] : i >= 0 }",
	  "{ [i] : exists (e0, e1: 3e1 >= 1 + 2e0 and "
	    "8e1 <= -1 + 5i - 5e0 and 2e1 >= 1 + 2i - 5e0) }" },
	{ TWO("{ [i=0:10] : exists a, b: 2b > 2i - 5a > 8b -3 i and 3b > 2a }"),
	  "{ [i=0:10] }",
	  "{ [i] : exists a, b: 2b > 2i - 5a > 8b -3 i and 3b > 2a }" },
	});

	C(&gist<isl::set> , {
	{ TWO("{ [1, -1, 3] }"),
	  "{ [1, b, 2 - b] : -1 <= b <= 2 }",
	  "{ [a, -1, c] }" },
	{ TWO("{ [a, b, c] : a <= 15 and a >= 1 }"),
	  "{ [a, b, c] : exists (e0 = floor((-1 + a)/16): a >= 1 and "
			"c <= 30 and 32e0 >= -62 + 2a + 2b - c and b >= 0) }",
	  "{ [a, b, c] : a <= 15 }" },
	{ TWO("{ : }"), "{ : 1 = 0 }", "{ : }" },
	{ TWO("{ : 1 = 0 }"), "{ : 1 = 0 }", "{ : }" },
	{ TWO("[M] -> { [x] : exists (e0 = floor((-2 + x)/3): 3e0 = -2 + x) }"),
	  "[M] -> { [3M] }" , "[M] -> { [x] : 1 = 0 }" },
	{ TWO("{ [m, n, a, b] : a <= 2147 + n }"),
	  "{ [m, n, a, b] : (m >= 1 and n >= 1 and a <= 2148 - m and "
			"b <= 2148 - n and b >= 0 and b >= 2149 - n - a) or "
			"(n >= 1 and a >= 0 and b <= 2148 - n - a and "
			"b >= 0) }",
	  "{ [m, n, ku, kl] }" },
	{ TWO("{ [a, a, b] : a >= 10 }"),
	  "{ [a, b, c] : c >= a and c <= b and c >= 2 }",
	  "{ [a, a, b] : a >= 10 }" },
	{ TWO("{ [i, j] : i >= 0 and i + j >= 0 }"), "{ [i, j] : i <= 0 }",
	  "{ [0, j] : j >= 0 }" },
	/* Check that no constraints on i6 are introduced in the gist */
	{ TWO("[t1] -> { [i4, i6] : exists (e0 = floor((1530 - 4t1 - 5i4)/20): "
		"20e0 <= 1530 - 4t1 - 5i4 and 20e0 >= 1511 - 4t1 - 5i4 and "
		"5e0 <= 381 - t1 and i4 <= 1) }"),
	  "[t1] -> { [i4, i6] : exists (e0 = floor((-t1 + i6)/5): "
		"5e0 = -t1 + i6 and i6 <= 6 and i6 >= 3) }",
	  "[t1] -> { [i4, i6] : exists (e0 = floor((1530 - 4t1 - 5i4)/20): "
		"i4 <= 1 and 5e0 <= 381 - t1 and 20e0 <= 1530 - 4t1 - 5i4 and "
		"20e0 >= 1511 - 4t1 - 5i4) }" },
	/* Check that no constraints on i6 are introduced in the gist */
	{ TWO("[t1, t2] -> { [i4, i5, i6] : exists (e0 = floor((1 + i4)/2), "
		"e1 = floor((1530 - 4t1 - 5i4)/20), "
		"e2 = floor((-4t1 - 5i4 + 10*floor((1 + i4)/2))/20), "
		"e3 = floor((-1 + i4)/2): t2 = 0 and 2e3 = -1 + i4 and "
			"20e2 >= -19 - 4t1 - 5i4 + 10e0 and 5e2 <= 1 - t1 and "
			"2e0 <= 1 + i4 and 2e0 >= i4 and "
			"20e1 <= 1530 - 4t1 - 5i4 and "
			"20e1 >= 1511 - 4t1 - 5i4 and i4 <= 1 and "
			"5e1 <= 381 - t1 and 20e2 <= -4t1 - 5i4 + 10e0) }"),
	  "[t1, t2] -> { [i4, i5, i6] : exists (e0 = floor((-17 + i4)/2), "
		"e1 = floor((-t1 + i6)/5): 5e1 = -t1 + i6 and "
			"2e0 <= -17 + i4 and 2e0 >= -18 + i4 and "
			"10e0 <= -91 + 5i4 + 4i6 and "
			"10e0 >= -105 + 5i4 + 4i6) }",
	  "[t1, t2] -> { [i4, i5, i6] : exists (e0 = floor((381 - t1)/5), "
		"e1 = floor((-1 + i4)/2): t2 = 0 and 2e1 = -1 + i4 and "
		"i4 <= 1 and 5e0 <= 381 - t1 and 20e0 >= 1511 - 4t1 - 5i4) }" },
	{ TWO("{ [0, 0, q, p] : -5 <= q <= 5 and p >= 0 }"),
	  "{ [a, b, q, p] : b >= 1 + a }",
	  "{ [a, b, q, p] : false }" },
	{ TWO("[n] -> { [x] : x = n && x mod 32 = 0 }"),
	  "[n] -> { [x] : x mod 32 = 0 }",
	  "[n] -> { [x = n] }" },
	{ TWO("{ [x] : x mod 6 = 0 }"), "{ [x] : x mod 3 = 0 }",
	  "{ [x] : x mod 2 = 0 }" },
	{ TWO("{ [x] : x mod 3200 = 0 }"), "{ [x] : x mod 10000 = 0 }",
	  "{ [x] : x mod 128 = 0 }" },
	{ TWO("{ [x] : x mod 3200 = 0 }"), "{ [x] : x mod 10 = 0 }",
	  "{ [x] : x mod 3200 = 0 }" },
	{ TWO("{ [a, b, c] : a mod 2 = 0 and a = c }"),
	  "{ [a, b, c] : b mod 2 = 0 and b = c }",
	  "{ [a, b, c = a] }" },
	{ TWO("{ [a, b, c] : a mod 6 = 0 and a = c }"),
	  "{ [a, b, c] : b mod 2 = 0 and b = c }",
	  "{ [a, b, c = a] : a mod 3 = 0 }" },
	{ TWO("{ [x] : 0 <= x <= 4 or 6 <= x <= 9 }"),
	  "{ [x] : 1 <= x <= 3 or 7 <= x <= 8 }",
	  "{ [x] }" },
	{ TWO("{ [x,y] : x < 0 and 0 <= y <= 4 or "
			"x >= -2 and -x <= y <= 10 + x }"),
	  "{ [x,y] : 1 <= y <= 3 }",
	  "{ [x,y] }" },
	});

	C(arg<isl::set>(&isl::pw_aff::gist), {
	{ "{ [x] -> [x] : x != 0 }", "{ [x] : x < -1 or x > 1 }",
	  "{ [x] -> [x] }" },
	});

	C(&isl::pw_aff::gist_params, {
	{ "[N] -> { D[x] -> [x] : N >= 0; D[x] -> [0] : N < 0 }",
	  "[N] -> { : N >= 0 }",
	  "[N] -> { D[x] -> [x] }" },
	});

	C(arg<isl::set>(&isl::multi_aff::gist), {
	{ "{ A[i] -> B[i, i] }", "{ A[0] }",
	  "{ A[i] -> B[0, 0] }" },
	{ "[N] -> { A[i] -> B[i, N] }", "[N] -> { A[0] : N = 5 }",
	  "[N] -> { A[i] -> B[0, 5] }" },
	{ "[N] -> { B[N + 1, N] }", "[N] -> { : N = 5 }",
	  "[N] -> { B[6, 5] }" },
	{ "[N] -> { A[i] -> B[] }", "[N] -> { A[0] : N = 5 }",
	  "[N] -> { A[i] -> B[] }" },
	{ "[N] -> { B[] }", "[N] -> { : N = 5 }",
	  "[N] -> { B[] }" },
	});

	C(&isl::multi_aff::gist_params, {
	{ "[N] -> { A[i] -> B[i, N] }", "[N] -> { : N = 5 }",
	  "[N] -> { A[i] -> B[i, 5] }" },
	{ "[N] -> { B[N + 1, N] }", "[N] -> { : N = 5 }",
	  "[N] -> { B[6, 5] }" },
	{ "[N] -> { A[i] -> B[] }", "[N] -> { : N = 5 }",
	  "[N] -> { A[i] -> B[] }" },
	{ "[N] -> { B[] }", "[N] -> { : N = 5 }",
	  "[N] -> { B[] }" },
	});

	C(arg<isl::set>(&isl::multi_pw_aff::gist), {
	{ "{ A[i] -> B[i, i] : i >= 0 }", "{ A[0] }",
	  "{ A[i] -> B[0, 0] }" },
	{ "[N] -> { A[i] -> B[i, N] : N >= 0 }", "[N] -> { A[0] : N = 5 }",
	  "[N] -> { A[i] -> B[0, 5] }" },
	{ "[N] -> { B[N + 1, N] }", "[N] -> { : N = 5 }",
	  "[N] -> { B[6, 5] }" },
	{ "[N] -> { A[i] -> B[] }", "[N] -> { A[0] : N = 5 }",
	  "[N] -> { A[i] -> B[] }" },
	{ "[N] -> { B[] }", "[N] -> { : N = 5 }",
	  "[N] -> { B[] }" },
	{ "{ A[i=0:10] -> B[i] }", "{ A[5] }",
	  "{ A[i] -> B[5] }" },
	{ "{ A[0:10] -> B[] }", "{ A[0:10] }",
	  "{ A[i] -> B[] }" },
	{ "[N] -> { A[i] -> B[] : N >= 0 }", "[N] -> { A[0] : N = 5 }",
	  "[N] -> { A[i] -> B[] }" },
	{ "[N] -> { B[] : N >= 0 }", "[N] -> { : N = 5 }",
	  "[N] -> { B[] }" },
	{ "[N] -> { B[] : N = 5 }", "[N] -> { : N >= 0 }",
	  "[N] -> { B[] : N = 5 }" },
	});

	C(&isl::multi_pw_aff::gist_params, {
	{ "[N] -> { A[i] -> B[i, N] : N >= 0 }", "[N] -> { : N = 5 }",
	  "[N] -> { A[i] -> B[i, 5] }" },
	{ "[N] -> { B[N + 1, N] }", "[N] -> { : N = 5 }",
	  "[N] -> { B[6, 5] }" },
	{ "[N] -> { A[i] -> B[] : N >= 0 }", "[N] -> { : N = 5 }",
	  "[N] -> { A[i] -> B[] }" },
	{ "[N] -> { B[] : N >= 0 }", "[N] -> { : N = 5 }",
	  "[N] -> { B[] }" },
	{ "[N] -> { B[] : N >= 5 }", "[N] -> { : N >= 0 }",
	  "[N] -> { B[] : N >= 5 }" },
	});

	C(&isl::multi_union_pw_aff::gist, {
	{ "C[{ B[i,i] -> [3i] }]", "{ B[i,i] }",
	  "C[{ B[i,j] -> [3i] }]" },
	{ "(C[] : { B[i,i] })", "{ B[i,i] }",
	  "(C[] : { B[i,j] })" },
	{ "[N] -> (C[] : { B[N,N] })", "[N] -> { B[N,N] }",
	  "[N] -> (C[] : { B[i,j] })" },
	{ "C[]", "{ B[i,i] }",
	  "C[]" },
	{ "[N] -> (C[] : { B[i,i] : N >= 0 })", "{ B[i,i] }",
	  "[N] -> (C[] : { B[i,j] : N >= 0 })" },
	{ "[N] -> (C[] : { : N >= 0 })", "{ B[i,i] }",
	  "[N] -> (C[] : { : N >= 0 })" },
	{ "[N] -> (C[] : { : N >= 0 })", "[N] -> { B[i,i] : N >= 0 }",
	  "[N] -> C[]" },
	});

	C(&isl::multi_union_pw_aff::gist_params, {
	{ "[N] -> C[{ B[i,i] -> [3i + N] }]", "[N] -> { : N = 1 }",
	  "[N] -> C[{ B[i,i] -> [3i + 1] }]" },
	{ "C[{ B[i,i] -> [3i] }]", "[N] -> { : N >= 0 }",
	  "[N] -> C[{ B[i,i] -> [3i] }]" },
	{ "[N] -> C[{ B[i,i] -> [3i] : N >= 0 }]", "[N] -> { : N >= 0 }",
	  "[N] -> C[{ B[i,i] -> [3i] }]" },
	{ "[N] -> C[{ B[i,i] -> [3i] : N >= 1 }]", "[N] -> { : N >= 0 }",
	  "[N] -> C[{ B[i,i] -> [3i] : N >= 1 }]" },
	{ "[N] -> (C[] : { B[i,i] : N >= 0 })", "[N] -> { : N >= 0 }",
	  "[N] -> (C[] : { B[i,i] })" },
	{ "[N] -> (C[] : { : N >= 0 })", "[N] -> { : N >= 0 }",
	  "[N] -> C[]" },
	{ "C[{ B[i,i] -> [3i] }]", "[N] -> { : N >= 0 }",
	  "[N] -> C[{ B[i,i] -> [3i] }]" },
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

/* Perform some basic reverse tests.
 */
static void test_reverse(isl::ctx ctx)
{
	C(&isl::aff::domain_reverse, {
	{ "{ T[A[] -> B[*]] -> [0] }",
	  "{ [B[*] -> A[]] -> [0] }" },
	{ "{ T[A[] -> A[]] -> [0] }",
	  "{ T[A[] -> A[]] -> [0] }" },
	{ "{ [A[x] -> B[y]] -> [5*(x // 2) + 7*(y // 3)] }",
	  "{ [B[y] -> A[x]] -> [5*(x // 2) + 7*(y // 3)] }" },
	});

	C(&isl::multi_aff::domain_reverse, {
	{ "{ [A[x] -> B[y]] -> [5*(x // 2) + 7*(y // 3)] }",
	  "{ [B[y] -> A[x]] -> [5*(x // 2) + 7*(y // 3)] }" },
	{ "{ [A[x] -> B[y]] -> T[5*(x // 2) + 7*(y // 3), 0] }",
	  "{ [B[y] -> A[x]] -> T[5*(x // 2) + 7*(y // 3), 0] }" },
	});

	C(&isl::set::wrapped_reverse, {
	{ "{ T[A[] -> B[*]] }",
	  "{ [B[*] -> A[]] }" },
	{ "{ T[A[] -> A[]] }",
	  "{ T[A[] -> A[]] }" },
	{ "{ [A[x] -> B[2x]] }",
	  "{ [B[y] -> A[x]] : y = 2x }" },
	});

	C(&isl::pw_aff::domain_reverse, {
	{ "{ [A[x] -> B[y]] -> [5*(x // 2) + 7*(y // 3)] }",
	  "{ [B[y] -> A[x]] -> [5*(x // 2) + 7*(y // 3)] }" },
	{ "{ [A[x] -> B[y]] -> [5*(x // 2) + 7*(y // 3)] : x > y }",
	  "{ [B[y] -> A[x]] -> [5*(x // 2) + 7*(y // 3)] : x > y }" },
	{ "{ [A[i] -> B[i + 1]] -> [i + 2] }",
	  "{ [B[i] -> A[i - 1]] -> [i + 1] }" },
	});

	C(&isl::pw_multi_aff::domain_reverse, {
	{ "{ [A[x] -> B[y]] -> T[5*(x // 2) + 7*(y // 3), 0] : x > y }",
	  "{ [B[y] -> A[x]] -> T[5*(x // 2) + 7*(y // 3), 0] : x > y }" },
	{ "{ [A[i] -> B[i + 1]] -> T[0, i + 2] }",
	  "{ [B[i] -> A[i - 1]] -> T[0, i + 1] }" },
	});

	C(&isl::multi_pw_aff::domain_reverse, {
	{ "{ [A[x] -> B[y]] -> T[5*(x // 2) + 7*(y // 3) : x > y, 0] }",
	  "{ [B[y] -> A[x]] -> T[5*(x // 2) + 7*(y // 3) : x > y, 0] }" },
	});

	C(&isl::map::domain_reverse, {
	{ "{ [A[] -> B[]] -> [C[] -> D[]] }",
	  "{ [B[] -> A[]] -> [C[] -> D[]] }" },
	{ "{ N[B[] -> C[]] -> A[] }",
	  "{ [C[] -> B[]] -> A[] }" },
	{ "{ N[B[x] -> B[y]] -> A[] }",
	  "{ N[B[*] -> B[*]] -> A[] }" },
	});

	C(&isl::union_map::domain_reverse, {
	{ "{ [A[] -> B[]] -> [C[] -> D[]] }",
	  "{ [B[] -> A[]] -> [C[] -> D[]] }" },
	{ "{ A[] -> [B[] -> C[]]; A[] -> B[]; A[0] -> N[B[1] -> B[2]] }",
	  "{ }" },
	{ "{ N[B[] -> C[]] -> A[] }",
	  "{ [C[] -> B[]] -> A[] }" },
	{ "{ N[B[x] -> B[y]] -> A[] }",
	  "{ N[B[*] -> B[*]] -> A[] }" },
	});

	C(&isl::union_map::range_reverse, {
	{ "{ A[] -> [B[] -> C[]]; A[] -> B[]; A[0] -> N[B[1] -> B[2]] }",
	  "{ A[] -> [C[] -> B[]]; A[0] -> N[B[2] -> B[1]] }" },
	{ "{ A[] -> N[B[] -> C[]] }",
	  "{ A[] -> [C[] -> B[]] }" },
	{ "{ A[] -> N[B[x] -> B[y]] }",
	  "{ A[] -> N[B[*] -> B[*]] }" },
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
	{ "lexmin", &test_lexmin },
	{ "gist", &test_gist },
	{ "project out parameters", &test_project },
	{ "reverse", &test_reverse },
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
