// Copyright John Maddock 2006, 2007.
// Copyright Paul A. Bristow 2007

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// test_cauchy.cpp Test Cauchy distribution

#ifdef _MSC_VER
#  pragma warning(disable: 4100) // unreferenced formal parameter.
// Seems an entirely spurious warning - formal parameter T IS used - get error if /* T */
//#  pragma warning(disable: 4535) // calling _set_se_translator() requires /EHa (in Boost.test)
// Enable C++ Exceptions Yes With SEH Exceptions (/EHa) prevents warning 4535.
#  pragma warning(disable: 4127) // conditional expression is constant
#endif

// #define BOOST_MATH_ASSERT_UNDEFINED_POLICY false 
// To compile even if Cauchy mean is used.
#include <boost/math/concepts/real_concept.hpp> // for real_concept
#include <boost/math/distributions/cauchy.hpp>
    using boost::math::cauchy_distribution;

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp> // Boost.Test
#include <boost/test/tools/floating_point_comparison.hpp>

#include "test_out_of_range.hpp"

#include <iostream>
   using std::cout;
   using std::endl;

template <class RealType>
void test_spots(RealType T)
{
  // Check some bad parameters to construct the distribution,
#ifndef BOOST_NO_EXCEPTIONS
  BOOST_CHECK_THROW(boost::math::cauchy_distribution<RealType> nbad1(0, 0), std::domain_error); // zero scale.
  BOOST_CHECK_THROW(boost::math::cauchy_distribution<RealType> nbad1(0, -1), std::domain_error); // negative scale (shape).
#else
  BOOST_CHECK_THROW(boost::math::cauchy_distribution<RealType>(0, 0), std::domain_error); // zero scale.
  BOOST_CHECK_THROW(boost::math::cauchy_distribution<RealType>(0, -1), std::domain_error); // negative scale (shape).
#endif
  cauchy_distribution<RealType> C01;

  BOOST_CHECK_EQUAL(C01.location(), 0); // Check standard values.
  BOOST_CHECK_EQUAL(C01.scale(), 1);

   // Basic sanity checks.
  // 50eps as a percentage, up to a maximum of double precision
  // (that's the limit of our test data).
  RealType tolerance = (std::max)(
     static_cast<RealType>(boost::math::tools::epsilon<double>()),
     boost::math::tools::epsilon<RealType>());
  tolerance *= 50 * 100;

  cout << "Tolerance for type " << typeid(T).name()  << " is " << tolerance << " %" << endl;

   // These first sets of test values were calculated by punching numbers
   // into a calculator, and using the formulas on the Mathworld website:
   // http://mathworld.wolfram.com/CauchyDistribution.html
   // and values from MathCAD 200 Professional, 
   // CDF:
   //
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.125)),              // x
         static_cast<RealType>(0.53958342416056554201085167134004L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-0.125)),              // x
         static_cast<RealType>(0.46041657583943445798914832865996L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.5)),              // x
         static_cast<RealType>(0.64758361765043327417540107622474L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-0.5)),              // x
         static_cast<RealType>(0.35241638234956672582459892377526L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(1.0)),              // x
         static_cast<RealType>(0.75),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-1.0)),              // x
         static_cast<RealType>(0.25),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(2.0)),              // x
         static_cast<RealType>(0.85241638234956672582459892377526L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-2.0)),              // x
         static_cast<RealType>(0.14758361765043327417540107622474L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(10.0)),              // x
         static_cast<RealType>(0.9682744825694464304850228813987L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-10.0)),              // x
         static_cast<RealType>(0.031725517430553569514977118601302L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-15000000.0)),
         static_cast<RealType>(0.000000021220659078919346664504384865488560725L),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      // Test the CDF at -max_value()/4.
      // For an input x of this magnitude, the reference value is 4/|x|/pi.
      ::boost::math::cdf(
         cauchy_distribution<RealType>(),
         -boost::math::tools::max_value<RealType>()/4),
         static_cast<RealType>(4)
                      / boost::math::tools::max_value<RealType>()
                      / boost::math::constants::pi<RealType>(),
         tolerance); // %

   //
   // Complements:
   //
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.125))),              // x
         static_cast<RealType>(0.46041657583943445798914832865996L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(-0.125))),              // x
         static_cast<RealType>(0.53958342416056554201085167134004L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.5))),              // x
         static_cast<RealType>(0.35241638234956672582459892377526L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(-0.5))),              // x
         static_cast<RealType>(0.64758361765043327417540107622474L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(1.0))),              // x
         static_cast<RealType>(0.25),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(-1.0))),              // x
         static_cast<RealType>(0.75),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(2.0))),              // x
         static_cast<RealType>(0.14758361765043327417540107622474L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(-2.0))),              // x
         static_cast<RealType>(0.85241638234956672582459892377526L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(10.0))),              // x
         static_cast<RealType>(0.031725517430553569514977118601302L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(-10.0))),              // x
         static_cast<RealType>(0.9682744825694464304850228813987L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(15000000.0))),
         static_cast<RealType>(0.000000021220659078919346664504384865488560725L),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      // Test the complemented CDF at max_value()/4.
      // For an input x of this magnitude, the reference value is 4/x/pi.
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(),
         boost::math::tools::max_value<RealType>()/4)),
         static_cast<RealType>(4)
                      / boost::math::tools::max_value<RealType>()
                      / boost::math::constants::pi<RealType>(),
         tolerance); // %

   //
   // Quantiles:
   //
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.53958342416056554201085167134004L)),
         static_cast<RealType>(0.125),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.46041657583943445798914832865996L)),
         static_cast<RealType>(-0.125),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.64758361765043327417540107622474L)),
         static_cast<RealType>(0.5),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.35241638234956672582459892377526)),
         static_cast<RealType>(-0.5),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.75)),
         static_cast<RealType>(1.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.25)),
         static_cast<RealType>(-1.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.85241638234956672582459892377526L)),
         static_cast<RealType>(2.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.14758361765043327417540107622474L)),
         static_cast<RealType>(-2.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.9682744825694464304850228813987L)),
         static_cast<RealType>(10.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.031725517430553569514977118601302L)),
         static_cast<RealType>(-10.0),
         tolerance); // %

   //
   // Quantile from complement:
   //
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.46041657583943445798914832865996L))),
         static_cast<RealType>(0.125),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.53958342416056554201085167134004L))),
         static_cast<RealType>(-0.125),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.35241638234956672582459892377526L))),
         static_cast<RealType>(0.5),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.64758361765043327417540107622474L))),
         static_cast<RealType>(-0.5),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.25))),
         static_cast<RealType>(1.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.75))),
         static_cast<RealType>(-1.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.14758361765043327417540107622474L))),
         static_cast<RealType>(2.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.85241638234956672582459892377526L))),
         static_cast<RealType>(-2.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.031725517430553569514977118601302L))),
         static_cast<RealType>(10.0),
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(),
         static_cast<RealType>(0.9682744825694464304850228813987L))),
         static_cast<RealType>(-10.0),
         tolerance); // %

   //
   // PDF
   //
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.125)),              // x
         static_cast<RealType>(0.31341281101173235351410956479511L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-0.125)),              // x
         static_cast<RealType>(0.31341281101173235351410956479511L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(0.5)),              // x
         static_cast<RealType>(0.25464790894703253723021402139602L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-0.5)),              // x
         static_cast<RealType>(0.25464790894703253723021402139602L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(1.0)),              // x
         static_cast<RealType>(0.15915494309189533576888376337251L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-1.0)),              // x
         static_cast<RealType>(0.15915494309189533576888376337251L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(2.0)),              // x
         static_cast<RealType>(0.063661977236758134307553505349006L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-2.0)),              // x
         static_cast<RealType>(0.063661977236758134307553505349006L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(10.0)),              // x
         static_cast<RealType>(0.0031515830315226799162155200667825L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(),
         static_cast<RealType>(-10.0)),              // x
         static_cast<RealType>(0.0031515830315226799162155200667825L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(2, 5),
         static_cast<RealType>(1)),              // x
         static_cast<RealType>(0.061213439650728975295724524374044L),                // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::pdf(
         cauchy_distribution<RealType>(-2, 0.25),
         static_cast<RealType>(1)),              // x
         static_cast<RealType>(0.0087809623774838805941453110826215L),                // probability.
         tolerance); // %

   //
   // The following test values were calculated using MathCad,
   // precision seems to be about 10^-13.
   //
   tolerance = (std::max)(tolerance, static_cast<RealType>(1e-11));
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(1, 1),
         static_cast<RealType>(0.125)),              // x
         static_cast<RealType>(0.271189304634946L),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(1, 1),
         static_cast<RealType>(0.125))),              // x
         static_cast<RealType>(1 - 0.271189304634946L),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(1, 1),
         static_cast<RealType>(0.271189304634946L)),              // x
         static_cast<RealType>(0.125),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(1, 1),
         static_cast<RealType>(1 - 0.271189304634946L))),              // x
         static_cast<RealType>(0.125),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(0.125)),              // x
         static_cast<RealType>(0.539583424160566L),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(0.5)),              // x
         static_cast<RealType>(0.647583617650433L),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(1)),              // x
         static_cast<RealType>(0.750000000000000),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(2)),              // x
         static_cast<RealType>(0.852416382349567),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(10)),              // x
         static_cast<RealType>(0.968274482569447),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(100)),              // x
         static_cast<RealType>(0.996817007235092),  // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(-0.125)),              // x
         static_cast<RealType>(0.460416575839434),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(-0.5)),              // x
         static_cast<RealType>(0.352416382349567),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(-1)),              // x
         static_cast<RealType>(0.2500000000000000),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(-2)),              // x
         static_cast<RealType>(0.147583617650433),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(-10)),              // x
         static_cast<RealType>(0.031725517430554),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(0, 1),
         static_cast<RealType>(-100)),              // x
         static_cast<RealType>(3.18299276490824E-3),  // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(1, 5),
         static_cast<RealType>(1.25)),              // x
         static_cast<RealType>(0.515902251256176),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(2, 2),
         static_cast<RealType>(1.25)),              // x
         static_cast<RealType>(0.385799748780092),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(4, 0.125),
         static_cast<RealType>(3)),              // x
         static_cast<RealType>(0.039583424160566),  // probability.
         tolerance); // % 
   BOOST_CHECK_CLOSE( 
      ::boost::math::cdf(
         cauchy_distribution<RealType>(-2, static_cast<RealType>(0.0001)),
         static_cast<RealType>(-3)),              // x
         static_cast<RealType>(3.1830988512275777e-5),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(4, 50),
         static_cast<RealType>(-3)),              // x
         static_cast<RealType>(0.455724386698215),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(-4, 50),
         static_cast<RealType>(-3)),              // x
         static_cast<RealType>(0.506365349100973),  // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(1, 5),
         static_cast<RealType>(1.25))),              // x
         static_cast<RealType>(1-0.515902251256176),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(2, 2),
         static_cast<RealType>(1.25))),              // x
         static_cast<RealType>(1-0.385799748780092),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(4, 0.125),
         static_cast<RealType>(3))),              // x
         static_cast<RealType>(1-0.039583424160566),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         cauchy_distribution<RealType>(-2, static_cast<RealType>(0.001)),
         static_cast<RealType>(-3)),              // x
         static_cast<RealType>(0.000318309780080539),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(4, 50),
         static_cast<RealType>(-3))),              // x
         static_cast<RealType>(1-0.455724386698215),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::cdf(
         complement(cauchy_distribution<RealType>(-4, 50),
         static_cast<RealType>(-3))),              // x
         static_cast<RealType>(1-0.506365349100973),  // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(1, 5),
         static_cast<RealType>(0.515902251256176)),              // x
         static_cast<RealType>(1.25),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(2, 2),
         static_cast<RealType>(0.385799748780092)),              // x
         static_cast<RealType>(1.25),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(4, 0.125),
         static_cast<RealType>(0.039583424160566)),              // x
         static_cast<RealType>(3),  // probability.
         tolerance); // %
   /*
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(-2, 0.0001),
         static_cast<RealType>(-3)),              // x
         static_cast<RealType>(0.000015915494296),  // probability.
         tolerance); // %
         */
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(4, 50),
         static_cast<RealType>(0.455724386698215)),              // x
         static_cast<RealType>(-3),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(-4, 50),
         static_cast<RealType>(0.506365349100973)),              // x
         static_cast<RealType>(-3),  // probability.
         tolerance); // %

   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(1, 5),
         static_cast<RealType>(1-0.515902251256176))),              // x
         static_cast<RealType>(1.25),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(2, 2),
         static_cast<RealType>(1-0.385799748780092))),              // x
         static_cast<RealType>(1.25),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(4, 0.125),
         static_cast<RealType>(1-0.039583424160566))),              // x
         static_cast<RealType>(3),  // probability.
         tolerance); // %
   /*
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         cauchy_distribution<RealType>(-2, 0.0001),
         static_cast<RealType>(-3)),              // x
         static_cast<RealType>(0.000015915494296),  // probability.
         tolerance); // %
         */
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(4, 50),
         static_cast<RealType>(1-0.455724386698215))),              // x
         static_cast<RealType>(-3),  // probability.
         tolerance); // %
   BOOST_CHECK_CLOSE(
      ::boost::math::quantile(
         complement(cauchy_distribution<RealType>(-4, 50),
         static_cast<RealType>(1-0.506365349100973))),              // x
         static_cast<RealType>(-3),  // probability.
         tolerance); // %

   cauchy_distribution<RealType> dist; // default (0, 1)
   BOOST_CHECK_EQUAL(
       mode(dist),
       static_cast<RealType>(0));
   BOOST_CHECK_EQUAL(
       median(dist),
       static_cast<RealType>(0));
   RealType expected_entropy = log(2*boost::math::constants::two_pi<RealType>());
   BOOST_CHECK_CLOSE(
       entropy(dist),
       expected_entropy, tolerance);

   //
   // Things that now don't compile (BOOST-STATIC_ASSERT_FAILURE) by default.
   // #define BOOST_MATH_ASSERT_UNDEFINED_POLICY false 
   // To compile even if Cauchy mean is used.
   // See policy reference, mathematically undefined function policies
   //
   //BOOST_CHECK_THROW(
   //    mean(dist),
   //    std::domain_error);
   //BOOST_CHECK_THROW(
   //    variance(dist),
   //    std::domain_error);
   //BOOST_CHECK_THROW(
   //    standard_deviation(dist),
   //    std::domain_error);
   //BOOST_CHECK_THROW(
   //    kurtosis(dist),
   //    std::domain_error);
   //BOOST_CHECK_THROW(
   //    kurtosis_excess(dist),
   //    std::domain_error);
   //BOOST_CHECK_THROW(
   //    skewness(dist),
   //    std::domain_error);

   BOOST_CHECK_THROW(
       quantile(dist, RealType(0.0)),
       std::overflow_error);
   BOOST_CHECK_THROW(
       quantile(dist, RealType(1.0)),
       std::overflow_error);
   BOOST_CHECK_THROW(
       quantile(complement(dist, RealType(0.0))),
       std::overflow_error);
   BOOST_CHECK_THROW(
       quantile(complement(dist, RealType(1.0))),
       std::overflow_error);

   check_out_of_range<boost::math::cauchy_distribution<RealType> >(0, 1); // (All) valid constructor parameter values.



} // template <class RealType>void test_spots(RealType)

BOOST_AUTO_TEST_CASE(test_main)
{
  BOOST_MATH_CONTROL_FP;
   // Check that can generate cauchy distribution using the two convenience methods:
  boost::math::cauchy mycd1(1.); // Using typedef
  cauchy_distribution<> mycd2(1.); // Using default RealType double.
  cauchy_distribution<> C01; // Using default RealType double for Standard Cauchy.
  BOOST_CHECK_EQUAL(C01.location(), 0); // Check standard values.
  BOOST_CHECK_EQUAL(C01.scale(), 1);

    // Basic sanity-check spot values.
   // (Parameter value, arbitrarily zero, only communicates the floating point type).
  test_spots(0.0F); // Test float. OK at decdigits = 0 tolerance = 0.0001 %
  test_spots(0.0); // Test double. OK at decdigits 7, tolerance = 1e07 %
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  test_spots(0.0L); // Test long double.
#if !BOOST_WORKAROUND(BOOST_BORLANDC, BOOST_TESTED_AT(0x582)) && !defined(BOOST_MATH_NO_REAL_CONCEPT_TESTS)
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
Output:

Running 1 test case...
Tolerance for type float is 0.000596046 %
Tolerance for type double is 1.11022e-012 %
Tolerance for type long double is 1.11022e-012 %
Tolerance for type class boost::math::concepts::real_concept is 1.11022e-012 %
*** No errors detected

*/
