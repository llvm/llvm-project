// test_find_scale.cpp

// Copyright John Maddock 2007.
// Copyright Paul A. Bristow 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Basic sanity test for find_scale function.

// Default distribution domain error policy is
// #define BOOST_MATH_DOMAIN_ERROR_POLICY throw_on_error

#include <pch.hpp>

#include <boost/math/tools/test.hpp>
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/distributions/normal.hpp> // for normal_distribution
  using boost::math::normal; // Default type double.
  using boost::math::normal_distribution; // All floating-point types.
#include <boost/math/distributions/cauchy.hpp> // for cauchy_distribution
  using boost::math::cauchy;
#include <boost/math/distributions/pareto.hpp> // for cauchy_distribution
  using boost::math::pareto;
#include <boost/math/distributions/find_scale.hpp>
  using boost::math::find_scale;
  using boost::math::complement;// will be needed by users who want complement,
#include <boost/math/policies/policy.hpp>
  using boost::math::policies::policy;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // for test_main
#include <boost/test/tools/floating_point_comparison.hpp> // for BOOST_CHECK_CLOSE_FRACTION, BOOST_CHECK_EQUAL...

#include <iostream>
#include <iomanip>
  using std::cout; using std::endl; using std::fixed;
  using std::right; using std::left; using std::showpoint;
  using std::showpos; using std::setw; using std::setprecision;

#include <limits>
  using std::numeric_limits;

template <class RealType> // Any floating-point type RealType.
void test_spots(RealType)
{ // Parameter only provides the type, float, double... value ignored.

  // Basic sanity checks, test data may be to double precision only
  // so set tolerance to 100 eps expressed as a fraction,
  // or 100 eps of type double expressed as a fraction,
  // whichever is the larger.

  RealType tolerance = (std::max)
      (boost::math::tools::epsilon<RealType>(),
      static_cast<RealType>(std::numeric_limits<double>::epsilon()));
   tolerance *= 100; // 100 eps as a fraction.

  cout << "Tolerance for type " << typeid(RealType).name()  << " is "
    << setprecision(3) << tolerance  << " (or " << tolerance * 100 << "%)." << endl;

  BOOST_MATH_CHECK_THROW( // Probability outside 0 to 1.
       find_scale<normal_distribution<RealType> >(
       static_cast<RealType>(0.), static_cast<RealType>(-1.), static_cast<RealType>(0.) ),
       std::domain_error);
  
  normal_distribution<RealType> n; // standard N(0,1)
  BOOST_CHECK_EQUAL(n.location(), 0); // aka mean.
  BOOST_CHECK_EQUAL(n.scale(), 1); // aka standard_deviation.

   // Check for 'bad' arguments.
  BOOST_MATH_CHECK_THROW(find_scale<normal>(0., -1., 0.), std::domain_error); // p below 0 to 1.
  BOOST_MATH_CHECK_THROW(find_scale<normal>(0., 2., 0.), std::domain_error); // p above 0 to 1.
  BOOST_MATH_CHECK_THROW(find_scale<normal>(numeric_limits<double>::infinity(), 0.5, 0.),
    std::domain_error); // z not finite.
  BOOST_MATH_CHECK_THROW(find_scale<normal>(numeric_limits<double>::quiet_NaN(), -1., 0.),
    std::domain_error); // z not finite
  BOOST_MATH_CHECK_THROW(find_scale<normal>(0., -1., numeric_limits<double>::quiet_NaN()),
    std::domain_error); // scale not finite


  BOOST_MATH_CHECK_THROW(find_scale<normal>(complement(0., -1., 0.)), std::domain_error); // p below 0 to 1.
  BOOST_MATH_CHECK_THROW(find_scale<normal>(complement(0., 2., 0.)), std::domain_error); // p above 0 to 1.
  BOOST_MATH_CHECK_THROW(find_scale<normal>(complement(numeric_limits<double>::infinity(), 0.5, 0.)),
    std::domain_error); // z not finite.
  BOOST_MATH_CHECK_THROW(find_scale<normal>(complement(numeric_limits<double>::quiet_NaN(), -1., 0.)),
    std::domain_error); // z not finite
  BOOST_MATH_CHECK_THROW(find_scale<normal>(complement(0., -1., numeric_limits<double>::quiet_NaN())),
    std::domain_error); // scale not finite

  BOOST_MATH_CHECK_THROW(find_scale<normal>(complement(0., -1., 0.)), std::domain_error); // p below 0 to 1.


  // Check for ab-use with unsuitable distribution(s), for example,
  // pareto distribution (and most others) can't be used with find_scale (or find_location)
  // because they lack the scale and location attributes.
  // BOOST_MATH_CHECK_THROW(find_scale<pareto>(0., 0.5, 0.), std::domain_error);
  // correctly fails to compile in find_scale() at
  // static_assert(::boost::math::tools::is_scaled_distribution<Dist>::value); 

  // Check doesn't throw when an ignore_error for domain_error policy is used.
  using boost::math::policies::policy;
  using boost::math::policies::domain_error;
  using boost::math::policies::ignore_error;

  // Define a (bad?) policy to ignore domain errors ('bad' arguments):
  typedef policy<domain_error<ignore_error> > ignore_domain_policy;
  // Using a typedef is convenient, especially if it is re-used.
#ifndef BOOST_NO_EXCEPTIONS
  BOOST_CHECK_NO_THROW(find_scale<normal>(0, -1, 1,
    ignore_domain_policy())); // probability outside [0, 1]
  BOOST_CHECK_NO_THROW(find_scale<normal>(numeric_limits<double>::infinity(), -1, 1,
    ignore_domain_policy())); // z not finite.
  BOOST_CHECK_NO_THROW(find_scale<normal>(complement(0, -1, 1, ignore_domain_policy()))); // probability outside [0, 1]
  BOOST_CHECK_NO_THROW(find_scale<normal>(complement(numeric_limits<double>::infinity(), -1, 1,
    ignore_domain_policy()))); // z not finite.
#endif
  RealType l = 0.; // standard normal distribution.
  RealType sd = static_cast<RealType>(1); // normal default standard deviation = 1.
  normal_distribution<RealType> n01(l, sd); // mean(location) = 0, standard_deviation (scale) = 1.
  RealType z = static_cast<RealType>(-2); // z to give prob p
  //cout << "Standard normal distribution with standard deviation = " << sd 
  //  << " has " << "fraction <= " << z << " = "  << cdf(n01, z) << endl;
  // Standard normal distribution with standard deviation = 1 has fraction <= -2 = 0.0227501

  //normal_distribution<RealType> np001pc(l, sd); // Same mean(location) but with standard_deviation (scale) changed.
  //cout << "Normal distribution with standard deviation = " << s 
  //  << " has " << "fraction <= " << z << " = "  << cdf(np001pc, z) << endl;

  // Find scale to give a probability p (0.001) of z (-2)
  RealType p = static_cast<RealType>(0.001); // only 0.1% to be below z (-2).
  // location (mean) remains at zero.
  RealType s = find_scale<normal_distribution<RealType> >(z, p, l);
  //cout << "Mean " << l << ", z " << z << ",  p " << p
  //  << ", sd " << sd << ", find_scale " << s 
  //  << ", difference in sd " << s - sd << endl;
  // Mean 0, z -2,  p 0.001, sd 1, find_scale 0.64720053440907599, difference in sd -0.352799

  cout.precision(17);
  BOOST_CHECK_CLOSE_FRACTION(s, static_cast<RealType>(0.64720053440907599L), tolerance);

  normal_distribution<RealType> np001pc(l, s); // Same mean(location) but with standard_deviation (scale) changed.
  //cout << "Normal distribution with standard deviation = " << s 
  //  << " has " << "fraction <= " << z << " = "  << cdf(np001pc, z) << endl;
  // Normal distribution with standard deviation = 0.647201 has fraction <= -2 = 0.001

  // Check cdf such that only fraction p really is below changed standard deviation s. 
  BOOST_CHECK_CLOSE_FRACTION(p, cdf(np001pc, z), tolerance);

  // Check that some policies can be applied (though results not used here).
  s = find_scale<normal_distribution<RealType> >(z, p, l, policy<>()); // Default policy, needs using boost::math::policies::policy;
  s = find_scale<normal_distribution<RealType> >(z, p, l, boost::math::policies::policy<>()); // Default policy, fully specified.
  s = find_scale<normal_distribution<RealType> >(z, p, l, ignore_domain_policy()); // find_scale with new policy, using typedef.
  s = find_scale<normal_distribution<RealType> >(z, p, l, policy<domain_error<ignore_error> >()); // New policy, without typedef.

  // Check that can use the complement version too.
  RealType q = 1 - p; // complement.
  s = find_scale<normal_distribution<RealType> >(complement(z, q, l)); // Implicit default policy.
  BOOST_CHECK_CLOSE_FRACTION(s, static_cast<RealType>(0.64720053440907599L), tolerance);
  s = find_scale<normal_distribution<RealType> >(complement(z, q, l, policy<>())); // Explicit default policy.
  BOOST_CHECK_CLOSE_FRACTION(s, static_cast<RealType>(0.64720053440907599L), tolerance);

  normal_distribution<RealType> np95pc(l, s); // Same mean(location) but with new standard_deviation (scale).

  //cout << "Mean " << l << ", z " << z << ",  q " << q
  //<< ", sd " << sd << ", find_scale " << s 
  //<< ", difference in sd " << s - sd << endl;

  //cout << "Normal distribution with standard deviation = " << s 
  //  << " has " << "fraction <= " << z << " = "  << cdf(np001pc, z) << endl;
  BOOST_CHECK_CLOSE_FRACTION(q, cdf(complement(np95pc, z)), tolerance);

} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE( test_main )
{
  // Basic sanity-check spot values.

   // (Parameter value, arbitrarily zero, only communicates the floating-point type).
  test_spots(0.0F); // Test float.
  test_spots(0.0); // Test double.
  #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test_spots(0.0L); // Test long double.
    #if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x0582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
    test_spots(boost::math::concepts::real_concept(0.)); // Test real concept.
  #endif
  #else
     std::cout << "<note>The long double tests have been disabled on this platform "
        "either because the long double overloads of the usual math functions are "
        "not available at all, or because they are too inaccurate for these tests "
        "to pass.</note>" << std::endl;
  #endif
  
} // BOOST_AUTO_TEST_CASE( test_main )

/*

Output is:

Autorun "i:\boost-06-05-03-1300\libs\math\test\Math_test\debug\test_find_scale.exe"
Running 1 test case...
Tolerance for type float is 1.19e-005 (or 0.00119%).
Tolerance for type double is 2.22e-014 (or 2.22e-012%).
Tolerance for type long double is 2.22e-014 (or 2.22e-012%).
Tolerance for type class boost::math::concepts::real_concept is 2.22e-014 (or 2.22e-012%).
*** No errors detected


*/


