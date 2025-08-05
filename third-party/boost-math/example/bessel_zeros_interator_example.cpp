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

//[bessel_zeros_iterator_example_1

/*`[h5 Using Output Iterator to sum zeros of Bessel Functions]

This example demonstrates summing zeros of the Bessel functions.
To use the functions for finding zeros of the functions we need
 */

#include <boost/math/special_functions/bessel.hpp>

/*`We use the `cyl_bessel_j_zero` output iterator parameter `out_it`
to create a sum of ['1/zeros[super 2]] by defining a custom output iterator:
*/

template <class T>
struct output_summation_iterator
{
   output_summation_iterator(T* p) : p_sum(p)
   {}
   output_summation_iterator& operator*()
   { return *this; }
    output_summation_iterator& operator++()
   { return *this; }
   output_summation_iterator& operator++(int)
   { return *this; }
   output_summation_iterator& operator = (T const& val)
   {
     *p_sum += 1./ (val * val); // Summing 1/zero^2.
     return *this;
   }
private:
   T* p_sum;
};

//] [/bessel_zeros_iterator_example_1]

int main()
{
  try
  {
//[bessel_zeros_iterator_example_2

/*`The sum is calculated for many values, converging on the analytical exact value of `1/8`.
*/
    using boost::math::cyl_bessel_j_zero;
    double nu = 1.;
    double sum = 0;
    output_summation_iterator<double> it(&sum);  // sum of 1/zeros^2
    cyl_bessel_j_zero(nu, 1, 10000, it);

    double s = 1/(4 * (nu + 1)); // 0.125 = 1/8 is exact analytical solution.
    std::cout << std::setprecision(6) << "nu = " << nu << ", sum = " << sum
      << ", exact = " << s << std::endl;
    // nu = 1.00000, sum = 0.124990, exact = 0.125000
//] [/bessel_zeros_iterator_example_2]
   }
  catch (std::exception const& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }
  return 0;
  } // int_main()

/*
 Output:

 nu = 1, sum = 0.12499, exact = 0.125
*/
