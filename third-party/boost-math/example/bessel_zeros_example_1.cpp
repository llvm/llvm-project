
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
// Test values can be calculated using [@wolframalpha.com WolframAlpha]
// See also http://dlmf.nist.gov/10.21

//[bessel_zeros_example_1

/*`This example demonstrates calculating zeros of the Bessel and Neumann functions.
It also shows how Boost.Math and Boost.Multiprecision can be combined to provide
a many decimal digit precision. For 50 decimal digit precision we need to include
*/

  #include <boost/multiprecision/cpp_dec_float.hpp>

/*`and a `typedef` for `float_type` may be convenient
(allowing a quick switch to re-compute at built-in `double` or other precision)
*/
  typedef boost::multiprecision::cpp_dec_float_50 float_type;

//`To use the functions for finding zeros of the functions we need

  #include <boost/math/special_functions/bessel.hpp>

//`This file includes the forward declaration signatures for the zero-finding functions:

//  #include <boost/math/special_functions/math_fwd.hpp>

/*`but more details are in the full documentation, for example at
[@http://www.boost.org/doc/libs/1_53_0/libs/math/doc/sf_and_dist/html/math_toolkit/special/bessel/bessel_over.html Boost.Math Bessel functions].
*/

/*`This example shows obtaining both a single zero of the Bessel function,
and then placing multiple zeros into a container like `std::vector` by providing an iterator.
*/
//] [/bessel_zeros_example_1]

/*The signature of the single value function is:

  template <class T>
  inline typename detail::bessel_traits<T, T, policies::policy<> >::result_type
    cyl_bessel_j_zero(
           T v,      // Floating-point value for Jv.
           int m);   // start index.

The result type is controlled by the floating-point type of parameter `v`
(but subject to the usual __precision_policy and __promotion_policy).

The signature of multiple zeros function is:

  template <class T, class OutputIterator>
  inline OutputIterator cyl_bessel_j_zero(
                                T v,                      // Floating-point value for Jv.
                                int start_index,          // 1-based start index.
                                unsigned number_of_zeros, // How many zeros to generate
                                OutputIterator out_it);   // Destination for zeros.

There is also a version which allows control of the __policy_section for error handling and precision.

  template <class T, class OutputIterator, class Policy>
  inline OutputIterator cyl_bessel_j_zero(
                                T v,                      // Floating-point value for Jv.
                                int start_index,          // 1-based start index.
                                unsigned number_of_zeros, // How many zeros to generate
                                OutputIterator out_it,    // Destination for zeros.
                                const Policy& pol);       // Policy to use.
*/

int main()
{
  try
  {
//[bessel_zeros_example_2

/*`[tip It is always wise to place code using Boost.Math inside try'n'catch blocks;
this will ensure that helpful error messages are shown when exceptional conditions arise.]

First, evaluate a single Bessel zero.

The precision is controlled by the float-point type of template parameter `T` of `v`
so this example has `double` precision, at least 15 but up to 17 decimal digits (for the common 64-bit double).
*/
//    double root = boost::math::cyl_bessel_j_zero(0.0, 1);
//    // Displaying with default precision of 6 decimal digits:
//    std::cout << "boost::math::cyl_bessel_j_zero(0.0, 1) " << root << std::endl; // 2.40483
//    // And with all the guaranteed (15) digits:
//    std::cout.precision(std::numeric_limits<double>::digits10);
//    std::cout << "boost::math::cyl_bessel_j_zero(0.0, 1) " << root << std::endl; // 2.40482555769577
/*`But note that because the parameter `v` controls the precision of the result,
`v` [*must be a floating-point type].
So if you provide an integer type, say 0, rather than 0.0, then it will fail to compile thus:
``
    root = boost::math::cyl_bessel_j_zero(0, 1);
``
with this error message
``
  error C2338: Order must be a floating-point type.
``

Optionally, we can use a policy to ignore errors, C-style, returning some value,
perhaps infinity or NaN, or the best that can be done. (See __user_error_handling).

To create a (possibly unwise!) policy `ignore_all_policy` that ignores all errors:
*/

  typedef boost::math::policies::policy<
    boost::math::policies::domain_error<boost::math::policies::ignore_error>,
    boost::math::policies::overflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::underflow_error<boost::math::policies::ignore_error>,
    boost::math::policies::denorm_error<boost::math::policies::ignore_error>,
    boost::math::policies::pole_error<boost::math::policies::ignore_error>,
    boost::math::policies::evaluation_error<boost::math::policies::ignore_error>
              > ignore_all_policy;
 //`Examples of use of this `ignore_all_policy` are

    double inf = std::numeric_limits<double>::infinity();
    double nan = std::numeric_limits<double>::quiet_NaN();

    double dodgy_root = boost::math::cyl_bessel_j_zero(-1.0, 1, ignore_all_policy());
    std::cout << "boost::math::cyl_bessel_j_zero(-1.0, 1) " << dodgy_root << std::endl; // 1.#QNAN
    double inf_root = boost::math::cyl_bessel_j_zero(inf, 1, ignore_all_policy());
    std::cout << "boost::math::cyl_bessel_j_zero(inf, 1) " << inf_root << std::endl; // 1.#QNAN
    double nan_root = boost::math::cyl_bessel_j_zero(nan, 1, ignore_all_policy());
    std::cout << "boost::math::cyl_bessel_j_zero(nan, 1) " << nan_root << std::endl; // 1.#QNAN

/*`Another version of `cyl_bessel_j_zero`  allows calculation of multiple zeros with one call,
placing the results in a container, often `std::vector`.
For example, generate and display the first five `double` roots of J[sub v] for integral order 2,
as column ['J[sub 2](x)] in table 1 of
[@ http://mathworld.wolfram.com/BesselFunctionZeros.html Wolfram Bessel Function Zeros].
*/
    unsigned int n_roots = 5U;
    std::vector<double> roots;
    boost::math::cyl_bessel_j_zero(2.0, 1, n_roots, std::back_inserter(roots));
    std::copy(roots.begin(),
              roots.end(),
              std::ostream_iterator<double>(std::cout, "\n"));

/*`Or we can use Boost.Multiprecision to generate 50 decimal digit roots of ['J[sub v]]
for non-integral order `v= 71/19 == 3.736842`, expressed as an exact-integer fraction
to generate the most accurate value possible for all floating-point types.

We set the precision of the output stream, and show trailing zeros to display a fixed 50 decimal digits.
*/
    std::cout.precision(std::numeric_limits<float_type>::digits10); // 50 decimal digits.
    std::cout << std::showpoint << std::endl; // Show trailing zeros.

    float_type x = float_type(71) / 19;
    float_type r = boost::math::cyl_bessel_j_zero(x, 1); // 1st root.
    std::cout << "x = " << x << ", r = " << r << std::endl;

    r = boost::math::cyl_bessel_j_zero(x, 20U); // 20th root.
    std::cout << "x = " << x << ", r = " << r << std::endl;

    std::vector<float_type> zeros;
    boost::math::cyl_bessel_j_zero(x, 1, 3, std::back_inserter(zeros));

    std::cout << "cyl_bessel_j_zeros" << std::endl;
    // Print the roots to the output stream.
    std::copy(zeros.begin(), zeros.end(),
              std::ostream_iterator<float_type>(std::cout, "\n"));
//] [/bessel_zeros_example_2]
  }
  catch (std::exception const& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }

 } // int main()

 /*

 Output:

   Description: Autorun "J:\Cpp\big_number\Debug\bessel_zeros_example_1.exe"
  boost::math::cyl_bessel_j_zero(-1.0, 1) 3.83171
  boost::math::cyl_bessel_j_zero(inf, 1) 1.#QNAN
  boost::math::cyl_bessel_j_zero(nan, 1) 1.#QNAN
  5.13562
  8.41724
  11.6198
  14.796
  17.9598
  
  x = 3.7368421052631578947368421052631578947368421052632, r = 7.2731751938316489503185694262290765588963196701623
  x = 3.7368421052631578947368421052631578947368421052632, r = 67.815145619696290925556791375555951165111460585458
  cyl_bessel_j_zeros
  7.2731751938316489503185694262290765588963196701623
  10.724858308883141732536172745851416647110749599085
  14.018504599452388106120459558042660282427471931581

*/

