
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

//[neumann_zeros_example_1

/*`[h5 Calculating zeros of the Neumann function.]
This example also shows how Boost.Math and Boost.Multiprecision can be combined to provide
a many decimal digit precision. For 50 decimal digit precision we need to include
*/

  #include <boost/multiprecision/cpp_dec_float.hpp>

/*`and a `typedef` for `float_type` may be convenient
(allowing a quick switch to re-compute at built-in `double` or other precision)
*/
  typedef boost::multiprecision::cpp_dec_float_50 float_type;

//`To use the functions for finding zeros of the `cyl_neumann` function we need:

  #include <boost/math/special_functions/bessel.hpp>
//] [/neumann_zerso_example_1]

int main()
{
  try
  {
    {
//[neumann_zeros_example_2
/*`The Neumann (Bessel Y) function zeros are evaluated very similarly:
*/
    using boost::math::cyl_neumann_zero;
    double zn = cyl_neumann_zero(2., 1);
    std::cout << "cyl_neumann_zero(2., 1) = " << zn << std::endl;

    std::vector<float> nzeros(3); // Space for 3 zeros.
    cyl_neumann_zero<float>(2.F, 1, nzeros.size(), nzeros.begin());

    std::cout << "cyl_neumann_zero<float>(2.F, 1, ";
    // Print the zeros to the output stream.
    std::copy(nzeros.begin(), nzeros.end(),
              std::ostream_iterator<float>(std::cout, ", "));

    std::cout << "\n""cyl_neumann_zero(static_cast<float_type>(220)/100, 1) = " 
      << cyl_neumann_zero(static_cast<float_type>(220)/100, 1) << std::endl;
    // 3.6154383428745996706772556069431792744372398748422

//] //[/neumann_zeros_example_2]
    }
  }
  catch (std::exception const& ex)
  {
    std::cout << "Thrown exception " << ex.what() << std::endl;
  }
} // int main()

/*
 Output:

cyl_neumann_zero(2., 1) = 3.38424
cyl_neumann_zero<float>(2.F, 1,
3.38424
6.79381
10.0235
3.61544
*/


