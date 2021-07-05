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

/* A description of the inputs and the output of a binary operation.
 */
struct binary {
	const char *arg1;
	const char *arg2;
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

/* A helper macro that calls test with as implicit initial argument "ctx" and
 * as extra argument a stringification of "FN".
 */
#define C(FN, ...) test(ctx, FN, #FN, __VA_ARGS__)

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

/* The list of tests to perform.
 */
static std::vector<std::pair<const char *, void (*)(isl::ctx)>> tests =
{
	{ "preimage", &test_preimage },
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
