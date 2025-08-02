// TestFactorial.cpp
//
// Factorials and Binomial Coefficients.
//
// Copyright Datasim Education BV 2009-2010
// Copyright John Maddock and Paul A. Bristow 2010

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/special_functions/factorials.hpp>
#include <boost/math/special_functions.hpp>

#include <iostream>
using namespace std;

int main()
{
  using namespace boost::math;

  // Factorials
  unsigned int n = 3;
 
  try
  {
     cout << "Factorial: " << factorial<double>(n) << endl;

     // Caution: You must provide a return type template value, so this will not compile
     // unsigned int nfac = factorial(n); // could not deduce template argument for 'T'
     // You must provide an explicit floating-point (not integer) return type.
     // If you do provide an integer type, like this:
     // unsigned int uintfac = factorial<unsigned int>(n);
     // you will also get a compile error, for MSVC C2338.
     // If you really want an integer type, you can convert from double:
     unsigned int intfac = static_cast<unsigned int>(factorial<double>(n));
     // this will be exact, until the result of the factorial overflows the integer type.

     cout << "Unchecked factorial: " << boost::math::unchecked_factorial<float>(n) << endl;
     // Note:
     // unsigned int unfac = boost::math::unchecked_factorial<unsigned int>(n);
     // also fails to compile for the same reasons.
  } 
  catch(exception& e)
  {
    cout << e.what() << endl;
  }

  // Double factorial n!!
  try
  {
    //cout << "Double factorial: " << boost::math::double_factorial<unsigned>(n);
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }

  // Rising and falling factorials
  try
  {
    int i = 2; double x = 8;
    cout << "Rising factorial: " << rising_factorial(x,i) << endl;
    cout << "Falling factorial: " << falling_factorial(x,i) << endl;
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }

  // Binomial coefficients
  try
  {
    unsigned n = 10; unsigned k = 2;
    // cout << "Binomial coefficient: " << boost::math::binomial_coefficient<unsigned>(n,k) << endl;
  }
  catch(exception& e)
  {
    cout << e.what() << endl;
  }
  return 0;
}

/*

Output:

  factorial_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\factorial_example.exe
  Factorial: 6
  Unchecked factorial: 6
  Rising factorial: 72
  Falling factorial: 56

*/


