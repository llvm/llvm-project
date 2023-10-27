#include <stdlib.h>

#include <exception>
#include <sstream>

#include <isl/options.h>
#include <isl/cpp.h>

/* Throw a runtime exception.
 */
static void die_impl(const char *file, int line, const char *message)
{
	std::ostringstream ss;
	ss << file << ":" << line << ": " << message;
	throw std::runtime_error(ss.str());
}

#define die(msg) die_impl(__FILE__, __LINE__, msg)

#include "isl_test_cpp17-generic.cc"

/* Check that an isl::exception_invalid gets thrown by "fn".
 */
static void check_invalid(const std::function<void(void)> &fn)
{
	bool caught = false;
	try {
		fn();
	} catch (const isl::exception_invalid &e) {
		caught = true;
	}
	if (!caught)
		die("no invalid exception was generated");
}

/* Test id::user.
 *
 * In particular, check that the object attached to an identifier
 * can be retrieved again and that retrieving an object of the wrong type
 * or retrieving an object when no object was attached results in an exception.
 */
static void test_user(isl::ctx ctx)
{
	isl::id id(ctx, "test", 5);
	isl::id id2(ctx, "test2");
	isl::id id3(ctx, "test3", std::string("s"));

	auto int_user = id.user<int>();
	if (int_user != 5)
		die("wrong integer retrieved from isl::id");
	auto s_user = id3.user<std::string>();
	if (s_user != "s")
		die("wrong string retrieved from isl::id");
	check_invalid([&id] () { id.user<std::string>(); });
	check_invalid([&id2] () { id2.user<int>(); });
	check_invalid([&id2] () { id2.user<std::string>(); });
	check_invalid([&id3] () { id3.user<int>(); });
}

/* Test the C++17 specific features of the (unchecked) isl C++ interface
 *
 * In particular, test
 *  - id::try_user
 *  - id::user
 */
int main()
{
	isl_ctx *ctx = isl_ctx_alloc();

	isl_options_set_on_error(ctx, ISL_ON_ERROR_ABORT);

	test_try_user(ctx);
	test_user(ctx);

	isl_ctx_free(ctx);

	return EXIT_SUCCESS;
}
