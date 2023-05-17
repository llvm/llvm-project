#include <stdlib.h>

#include <exception>
#include <iostream>

#include <isl/options.h>
#include <isl/cpp-checked.h>

/* Select the "checked" interface.
 */
namespace isl { using namespace checked; }

/* Print an error message and abort.
 */
static void die_impl(const char *file, int line, const char *message)
{
	std::cerr << file << ":" << line << ": " << message << "\n";
	exit(EXIT_FAILURE);
}

#define die(msg) die_impl(__FILE__, __LINE__, msg)

#include "isl_test_cpp17-generic.cc"

/* Test the C++17 specific features of the isl checked C++ interface
 *
 * In particular, test
 *  - id::try_user
 */
int main()
{
	isl_ctx *ctx = isl_ctx_alloc();

	isl_options_set_on_error(ctx, ISL_ON_ERROR_ABORT);

	test_try_user(ctx);

	isl_ctx_free(ctx);

	return EXIT_SUCCESS;
}
