
// Copyright Christopher Kormanyos 2013.
// Copyright Paul A. Bristow 2013.
// Copyright John Maddock 2013.

// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt or
// copy at http://www.boost.org/LICENSE_1_0.txt).

#ifdef _MSC_VER
#  pragma warning (disable : 4512) // assignment operator could not be generated.
#  pragma warning (disable : 4996) // assignment operator could not be generated.
#endif

#include <iostream>
#include <limits>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <iterator>

// Weisstein, Eric W. "Bessel Function Zeros." From MathWorld--A Wolfram Web Resource.
// http://mathworld.wolfram.com/BesselFunctionZeros.html
// Test values can be calculated using [@wolframalpha.com WolframAplha]
// See also http://dlmf.nist.gov/10.21

//[airy_zeros_example_1

/*`This example demonstrates calculating zeros of the Airy functions.
It also shows how Boost.Math and Boost.Multiprecision can be combined to provide
a many decimal digit precision. For 50 decimal digit precision we need to include
*/

  #include <boost/multiprecision/cpp_dec_float.hpp>

/*`and a `typedef` for `float_type` may be convenient
(allowing a quick switch to re-compute at built-in `double` or other precision)
*/
  typedef boost::multiprecision::cpp_dec_float_50 float_type;

//`To use the functions for finding zeros of the functions we need

  #include <boost/math/special_functions/airy.hpp>

/*`This example shows obtaining both a single zero of the Airy functions,
and then placing multiple zeros into a container like `std::vector` by providing an iterator.
The signature of the single-value Airy Ai function is:

  template <class T>
  T airy_ai_zero(unsigned m); // 1-based index of the zero.

The signature of multiple zeros Airy Ai function is:

  template <class T, class OutputIterator>
  OutputIterator airy_ai_zero(
                           unsigned start_index, // 1-based index of the zero.
                           unsigned number_of_zeros, // How many zeros to generate.
                           OutputIterator out_it); // Destination for zeros.

There are also versions which allows control of the __policy_section for error handling and precision.

  template <class T, class OutputIterator, class Policy>
  OutputIterator airy_ai_zero(
                           unsigned start_index, // 1-based index of the zero.
                           unsigned number_of_zeros, // How many zeros to generate.
                           OutputIterator out_it, // Destination for zeros.
                           const Policy& pol);  // Policy to use.
*/
//] [/airy_zeros_example_1]

int main()
{
  try
  {
//[airy_zeros_example_2

/*`[tip It is always wise to place code using Boost.Math inside `try'n'catch` blocks;
this will ensure that helpful error messages are shown when exceptional conditions arise.]

First, evaluate a single Airy zero.

The precision is controlled by the template parameter `T`,
so this example has `double` precision, at least 15 but up to 17 decimal digits
(for the common 64-bit double).
*/
    double aiz1 = boost::math::airy_ai_zero<double>(1);
    std::cout << "boost::math::airy_ai_zero<double>(1) = " << aiz1 << std::endl; 
    double aiz2 = boost::math::airy_ai_zero<double>(2);
    std::cout << "boost::math::airy_ai_zero<double>(2) = " << aiz2 << std::endl;
    double biz3 = boost::math::airy_bi_zero<double>(3);
    std::cout << "boost::math::airy_bi_zero<double>(3) = " << biz3 << std::endl;

/*`Other versions of `airy_ai_zero` and `airy_bi_zero`
allow calculation of multiple zeros with one call,
placing the results in a container, often `std::vector`.
For example, generate and display the first five `double` roots
[@http://mathworld.wolfram.com/AiryFunctionZeros.html Wolfram Airy Functions Zeros].
*/
    unsigned int n_roots = 5U;
    std::vector<double> roots;
    boost::math::airy_ai_zero<double>(1U, n_roots, std::back_inserter(roots));
    std::cout << "airy_ai_zeros:" << std::endl;
    std::copy(roots.begin(),
              roots.end(),
              std::ostream_iterator<double>(std::cout, "\n"));

/*`The first few real roots of Ai(x) are approximately -2.33811, -4.08795, -5.52056, -6.7867144, -7.94413, -9.02265 ...

Or we can use Boost.Multiprecision to generate 50 decimal digit roots.

We set the precision of the output stream, and show trailing zeros to display a fixed 50 decimal digits.
*/
    std::cout.precision(std::numeric_limits<float_type>::digits10); // float_type has 50 decimal digits.
    std::cout << std::showpoint << std::endl; // Show trailing zeros too.

    unsigned int m = 1U;
    float_type r = boost::math::airy_ai_zero<float_type>(1U); // 1st root.
    std::cout << "boost::math::airy_bi_zero<float_type>(" << m << ")  = " << r << std::endl;
    m = 2;
    r = boost::math::airy_ai_zero<float_type>(2U); // 2nd root.
    std::cout << "boost::math::airy_bi_zero<float_type>(" << m << ")  = " << r << std::endl;
    m = 7U;
    r = boost::math::airy_bi_zero<float_type>(7U); // 7th root.
    std::cout << "boost::math::airy_bi_zero<float_type>(" << m << ")  = " << r << std::endl;

    std::vector<float_type> zeros;
    boost::math::airy_ai_zero<float_type>(1U, 3, std::back_inserter(zeros));
    std::cout << "airy_ai_zeros:" << std::endl;
    // Print the roots to the output stream.
    std::copy(zeros.begin(), zeros.end(),
              std::ostream_iterator<float_type>(std::cout, "\n"));
//] [/airy_zeros_example_2]
  }
  catch (std::exception ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

 } // int main()

/*

 Output:

  Description: Autorun "J:\Cpp\big_number\Debug\airy_zeros_example.exe"
  boost::math::airy_ai_zero<double>(1) = -2.33811
  boost::math::airy_ai_zero<double>(2) = -4.08795
  boost::math::airy_bi_zero<double>(3) = -4.83074
  airy_ai_zeros:
  -2.33811
  -4.08795
  -5.52056
  -6.78671
  -7.94413

  boost::math::airy_bi_zero<float_type>(1)  = -2.3381074104597670384891972524467354406385401456711
  boost::math::airy_bi_zero<float_type>(2)  = -4.0879494441309706166369887014573910602247646991085
  boost::math::airy_bi_zero<float_type>(7)  = -9.5381943793462388866329885451560196208390720763825
  airy_ai_zeros:
  -2.3381074104597670384891972524467354406385401456711
  -4.0879494441309706166369887014573910602247646991085
  -5.5205598280955510591298555129312935737972142806175

*/

