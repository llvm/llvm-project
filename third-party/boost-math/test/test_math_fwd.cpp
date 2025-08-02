// test_math_fwd.cpp

//  Copyright John Maddock 2010.
//  Copyright Paul A. Bristow 2010.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

// Basic sanity check that special functions forward declaration header
// <boost/math/special_functions/math_fwd.hpp>
// and distributions forward declarations header
// <boost/math/distributions/fwd.hpp>
// #includes all the files that it needs to.
//
#include <boost/math/special_functions/math_fwd.hpp>
#include <boost/math/special_functions/beta.hpp>
// using boost::math::beta;

#include <boost/math/distributions/fwd.hpp>
#include <boost/math/distributions/normal.hpp>
// using boost::math::normal_distribution;

int main()
{
  // Special functions.
  // Call functions, discarding any result.
  using boost::math::beta;
  beta(1.,2.);

  // Distributions.
  using boost::math::normal_distribution;
  using boost::math::normal;

  // Construct some distributions.
  normal myf1(1., 2); // Using typedef.
  normal n01; // Use default values for mean and standard deviation).
  normal_distribution<> n01d(1., 2); // Using default RealType double.
  normal_distribution<float> n01f; // Using float type, and defaults.
  normal_distribution<float> myf22(0.f, 2.f); // Using explicit RealType float.

  return 0;
}

/*

VS2010

------ Build started: Project: test_math_fwd, Configuration: Debug Win32 ------
  test_math_fwd.cpp
  test_math_fwd.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Debug\test_math_fwd.exe
========== Build: 1 succeeded, 0 failed, 0 up-to-date, 0 skipped ==========


------ Build started: Project: test_math_fwd, Configuration: Release Win32 ------
  test_math_fwd.cpp
  Generating code
  Finished generating code
  test_math_fwd.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\test_math_fwd.exe

*/


