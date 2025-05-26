/*
 * Copyright Evan Miller, 2020
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#include <pch_light.hpp>
#include <boost/math/concepts/real_concept.hpp>
#include "test_jacobi_theta.hpp"

// Test file for the Jacobi Theta functions, a.k.a the four horsemen of the
// Jacobi elliptic integrals. At the moment only Wolfrma Alpha spot checks are
// used. We should generate extra-precise numbers with NTL::RR or some such.

void expected_results()
{
   //
   // Define the max and mean errors expected for
   // various compilers and platforms.
   //
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                  // test type(s)
      ".*Small Tau.*",      // test data group
      ".*", 1000, 200);  // test function

   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                  // test type(s)
      ".*WolframAlpha.*",      // test data group
      ".*", 60, 15);  // test function

   // Catch all cases come last:
   //
   add_expected_result(
      ".*",                          // compiler
      ".*",                          // stdlib
      ".*",                          // platform
      ".*",                  // test type(s)
      ".*",      // test data group
      ".*", 20, 5);  // test function
   //
   // Finish off by printing out the compiler/stdlib/platform names,
   // we do this to make it easier to mark up expected error rates.
   //
   std::cout << "Tests run with " << BOOST_COMPILER << ", "
      << BOOST_STDLIB << ", " << BOOST_PLATFORM << std::endl;
}

BOOST_AUTO_TEST_CASE( test_main )
{
    expected_results();
    BOOST_MATH_CONTROL_FP;
    BOOST_MATH_STD_USING

    using namespace boost::math;

    BOOST_CHECK_THROW(jacobi_theta1(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta1(0.0, 1.0), std::domain_error);

    BOOST_CHECK_THROW(jacobi_theta2(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta2(0.0, 1.0), std::domain_error);

    BOOST_CHECK_THROW(jacobi_theta3(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta3(0.0, 1.0), std::domain_error);

    BOOST_CHECK_THROW(jacobi_theta4(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta4(0.0, 1.0), std::domain_error);

    BOOST_CHECK_THROW(jacobi_theta1tau(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta1tau(0.0, -1.0), std::domain_error);

    BOOST_CHECK_THROW(jacobi_theta2tau(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta2tau(0.0, -1.0), std::domain_error);

    BOOST_CHECK_THROW(jacobi_theta3tau(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta3tau(0.0, -1.0), std::domain_error);

    BOOST_CHECK_THROW(jacobi_theta4tau(0.0, 0.0), std::domain_error);
    BOOST_CHECK_THROW(jacobi_theta4tau(0.0, -1.0), std::domain_error);

    double eps = std::numeric_limits<double>::epsilon();
    for (double q=0.0078125; q<1.0; q += 0.0078125) { // = 1/128
        for (double z=-8.0; z<=8.0; z += 0.125) {
            test_periodicity(z, q, 100 * eps);
            test_argument_translation(z, q, 100 * eps);
            test_sums_of_squares(z, q, 100 * eps);
            // The addition formula is complicated, cut it some extra slack
            test_addition_formulas(z, constants::ln_two<double>(), q, sqrt(sqrt(eps)));
            test_duplication_formula(z, q, 100 * eps);
            test_transformations_of_nome(z, q, 100 * eps);
            test_watsons_identities(z, 0.5, q, 101 * eps);
            test_landen_transformations(z, -log(q)/constants::pi<double>(), sqrt(eps));
            test_elliptic_functions(z, q, 5 * sqrt(eps));
        }
        test_elliptic_integrals(q, 10 * eps);
    }

    test_special_values(eps);

    for (double s=0.125; s<3.0; s+=0.125) {
        test_mellin_transforms(2.0 + s, eps, 3 * eps);
        test_laplace_transforms(s, eps, 4 * eps);
    }

    test_spots(0.0F, "float");
    test_spots(0.0, "double");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_spots(0.0L, "long double");
#ifndef BOOST_MATH_NO_REAL_CONCEPT_TESTS
    test_spots(concepts::real_concept(0), "real_concept");
#endif
#else
   std::cout << "<note>The long double tests have been disabled on this platform "
      "either because the long double overloads of the usual math functions are "
      "not available at all, or because they are too inaccurate for these tests "
      "to pass.</note>" << std::endl;
#endif
}
