// error_policies_example.cpp

// Copyright Paul A. Bristow 2007, 2010.
// Copyright John Maddock 2007.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/math/distributions/normal.hpp>
  using boost::math::normal_distribution;

#include <boost/math/distributions/students_t.hpp>
   using boost::math::students_t;  // Probability of students_t(df, t).
   using boost::math::students_t_distribution;

//  using namespace boost::math; causes:
//.\error_policy_normal.cpp(30) : error C2872: 'policy' : ambiguous symbol
//        could be '\boost/math/policies/policy.hpp(392) : boost::math::policies::policy'
//        or 'boost::math::policies'

// So should not use this 'using namespace boost::math;' command.

// Suppose we want a statistical distribution to return infinities,
// rather than throw exceptions (the default policy), then we can use:

// std
#include <iostream>
   using std::cout;
   using std::endl;

// using namespace boost::math::policies; or

using boost::math::policies::policy;
// Possible errors
using boost::math::policies::overflow_error;
using boost::math::policies::underflow_error;
using boost::math::policies::domain_error;
using boost::math::policies::pole_error;
using boost::math::policies::denorm_error;
using boost::math::policies::evaluation_error;
using boost::math::policies::ignore_error;

// Define a custom policy to ignore just overflow:
typedef policy<
overflow_error<ignore_error>
      > my_policy;

// Define another custom policy (perhaps ill-advised?)
// to ignore all errors: domain, pole, overflow, underflow, denorm & evaluation:
typedef policy<
domain_error<ignore_error>,
pole_error<ignore_error>,
overflow_error<ignore_error>,
underflow_error<ignore_error>,
denorm_error<ignore_error>,
evaluation_error<ignore_error>
      > my_ignoreall_policy;

// Define a new distribution with a custom policy to ignore_error
// (& thus perhaps return infinity for some arguments):
typedef boost::math::normal_distribution<double, my_policy> my_normal;
// Note: uses default parameters zero mean and unit standard deviation.

// We could also do the same for another distribution, for example:
using boost::math::students_t_distribution;
typedef students_t_distribution<double, my_ignoreall_policy> my_students_t;

int main()
{
  cout << "quantile(my_normal(), 0.05); = " << quantile(my_normal(), 0.05) << endl; // 0.05 is argument within normal range.
  cout << "quantile(my_normal(), 0.); = " << quantile(my_normal(), 0.) << endl; // argument zero, so expect infinity.
  cout << "quantile(my_normal(), 0.); = " << quantile(my_normal(), 0.F) << endl; // argument zero, so expect infinity.

  cout << "quantile(my_students_t(), 0.); = " << quantile(my_students_t(-1), 0.F) << endl; // 'bad' argument negative, so expect NaN.

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
  // Construct a (0, 1) normal distribution that ignores all errors,
  // returning NaN, infinity, zero, or best guess,
  // and NOT setting errno.
  normal_distribution<long double, my_ignoreall_policy> my_normal2(0.L, 1.L); // explicit parameters for distribution.
  cout << "quantile(my_normal2(), 0.); = " << quantile(my_normal2, 0.01) << endl; // argument 0.01, so result finite.
  cout << "quantile(my_normal2(), 0.); = " << quantile(my_normal2, 0.) << endl; // argument zero, so expect infinity.
#endif

  return 0;
}

/*

Output:

error_policies_example.cpp
  Generating code
  Finished generating code
  error_policy_normal_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Release\error_policies_example.exe
  quantile(my_normal(), 0.05); = -1.64485
  quantile(my_normal(), 0.); = -1.#INF
  quantile(my_normal(), 0.); = -1.#INF
  quantile(my_students_t(), 0.); = 1.#QNAN
  quantile(my_normal2(), 0.); = -2.32635
  quantile(my_normal2(), 0.); = -1.#INF

*/
