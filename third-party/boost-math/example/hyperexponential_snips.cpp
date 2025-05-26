// Copyright John Maddock 2014.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Caution: this file contains Quickbook markup as well as code
// and comments, don't change any of the special comment markups!

#ifdef _MSC_VER
#  pragma warning (disable : 4996) // disable -D_SCL_SECURE_NO_WARNINGS C++ 'Checked Iterators'
#endif

#include <boost/math/distributions/hyperexponential.hpp>
#include <iostream>

#ifndef BOOST_NO_CXX11_HDR_ARRAY
#include <array>
#endif

int main()
{
   {
//[hyperexponential_snip1
//=#include <boost/math/distributions/hyperexponential.hpp>
//=#include <iostream>
//=int main()
//={
   const double rates[] = { 1.0 / 10.0, 1.0 / 12.0 };

   boost::math::hyperexponential he(rates);

   std::cout << "Average lifetime: "
      << boost::math::mean(he)
      << " years" << std::endl;
   std::cout << "Probability that the appliance will work for more than 15 years: "
      << boost::math::cdf(boost::math::complement(he, 15.0))
      << std::endl;
//=}
//]
   }
   using namespace boost::math;
#ifndef BOOST_NO_CXX11_HDR_ARRAY
   {
   //[hyperexponential_snip2
   std::array<double, 2> phase_prob = { 0.5, 0.5 };
   std::array<double, 2> rates = { 1.0 / 10, 1.0 / 12 };

   hyperexponential he(phase_prob.begin(), phase_prob.end(), rates.begin(), rates.end());
   //]
   }

   {
   //[hyperexponential_snip3
   // We could be using any standard library container here... vector, deque, array, list etc:
   std::array<double, 2> phase_prob = { 0.5, 0.5 };
   std::array<double, 2> rates      = { 1.0 / 10, 1.0 / 12 };

   hyperexponential he1(phase_prob, rates);    // Construct from standard library container.

   double phase_probs2[] = { 0.5, 0.5 };
   double rates2[]       = { 1.0 / 10, 1.0 / 12 };

   hyperexponential he2(phase_probs2, rates2);  // Construct from native C++ array.
   //]
   }
   {
   //[hyperexponential_snip4
   // We could be using any standard library container here... vector, deque, array, list etc:
   std::array<double, 2> rates = { 1.0 / 10, 1.0 / 12 };

   hyperexponential he(rates.begin(), rates.end());

   BOOST_MATH_ASSERT(he.probabilities()[0] == 0.5); // Phase probabilities will be equal and normalised to unity.
   //]
   }
   {
   //[hyperexponential_snip5
   std::array<double, 2> rates = { 1.0 / 10, 1.0 / 12 };

   hyperexponential he(rates);

   BOOST_MATH_ASSERT(he.probabilities()[0] == 0.5); // Phase probabilities will be equal and normalised to unity.
   //]
   }
#endif
#if !defined(BOOST_NO_CXX11_HDR_INITIALIZER_LIST) && !(defined(BOOST_GCC_VERSION) && (BOOST_GCC_VERSION < 40500))
   {
   //[hyperexponential_snip6
   hyperexponential he = { { 0.5, 0.5 }, { 1.0 / 10, 1.0 / 12 } };
   //]
   }
   {
   //[hyperexponential_snip7
   hyperexponential he = { 1.0 / 10, 1.0 / 12 };

   BOOST_MATH_ASSERT(he.probabilities()[0] == 0.5);
   //]
   }
#endif
   return 0;
}
