/*
 * Copyright John Maddock, 2017
 * Use, modification and distribution are subject to the
 * Boost Software License, Version 1.0. (See accompanying file
 * LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 *
 * This example Illustrates numerical integration via Gauss and Gauss-Kronrod quadrature.
 */

#include <iostream>
#include <cmath>
#include <limits>
#include <boost/math/quadrature/gauss.hpp>
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>

void gauss_examples()
{
   //[gauss_example

   /*`
   We'll begin by integrating t[super 2] atan(t) over (0,1) using a 7 term Gauss-Legendre rule,
   and begin by defining the function to integrate as a C++ lambda expression:
   */
   using namespace boost::math::quadrature;

   auto f = [](const double& t) { return t * t * std::atan(t); };

   /*`
   Integration is simply a matter of calling the `gauss<double, 7>::integrate` method:
   */

   double Q = gauss<double, 7>::integrate(f, 0, 1);

   /*`
   Which yields a value 0.2106572512 accurate to 1e-10.

   For more accurate evaluations, we'll move to a multiprecision type and use a 20-point integration scheme:
   */

   using boost::multiprecision::cpp_bin_float_quad;

   auto f2 = [](const cpp_bin_float_quad& t) { return t * t * atan(t); };

   cpp_bin_float_quad Q2 = gauss<cpp_bin_float_quad, 20>::integrate(f2, 0, 1);

   /*`
   Which yields 0.2106572512258069881080923020669, which is accurate to 5e-28.
   */

   //]

   std::cout << std::setprecision(18) << Q << std::endl;
   std::cout << boost::math::relative_difference(Q, (boost::math::constants::pi<double>() - 2 + 2 * boost::math::constants::ln_two<double>()) / 12) << std::endl;

   std::cout << std::setprecision(34) << Q2 << std::endl;
   std::cout << boost::math::relative_difference(Q2, (boost::math::constants::pi<cpp_bin_float_quad>() - 2 + 2 * boost::math::constants::ln_two<cpp_bin_float_quad>()) / 12) << std::endl;
}

void gauss_kronrod_examples()
{
   //[gauss_kronrod_example

   /*`
   We'll begin by integrating exp(-t[super 2]/2) over (0,+[infin]) using a 7 term Gauss rule
   and 15 term Kronrod rule,
   and begin by defining the function to integrate as a C++ lambda expression:
   */
   using namespace boost::math::quadrature;

   auto f1 = [](double t) { return std::exp(-t*t / 2); };

   //<-
   double Q_expected = sqrt(boost::math::constants::half_pi<double>());
   //->

   /*`
   We'll start off with a one shot (ie non-adaptive)
   integration, and keep track of the estimated error:
   */
   double error;
   double Q = gauss_kronrod<double, 15>::integrate(f1, 0, std::numeric_limits<double>::infinity(), 0, 0, &error);

   /*`
   This yields Q = 1.25348207361, which has an absolute error of 1e-4 compared to the estimated error
   of 5e-3: this is fairly typical, with the difference between Gauss and Gauss-Kronrod schemes being
   much higher than the actual error.  Before moving on to adaptive quadrature, lets try again
   with more points, in fact with the largest Gauss-Kronrod scheme we have cached (30/61):
   */
   //<-
   std::cout << std::setprecision(16) << Q << std::endl;
   std::cout << boost::math::relative_difference(Q, Q_expected) << std::endl;
   std::cout << fabs(Q - Q_expected) << std::endl;
   std::cout << error << std::endl;
   //->
   Q = gauss_kronrod<double, 61>::integrate(f1, 0, std::numeric_limits<double>::infinity(), 0, 0, &error);
   //<-
   std::cout << std::setprecision(16) << Q << std::endl;
   std::cout << boost::math::relative_difference(Q, Q_expected) << std::endl;
   std::cout << fabs(Q - Q_expected) << std::endl;
   std::cout << error << std::endl;
   //->
   /*`
   This yields an absolute error of 3e-15 against an estimate of 1e-8, which is about as good as we're going to get
   at double precision

   However, instead of continuing with ever more points, lets switch to adaptive integration, and set the desired relative
   error to 1e-14 against a maximum depth of 5:
   */
   Q = gauss_kronrod<double, 15>::integrate(f1, 0, std::numeric_limits<double>::infinity(), 5, 1e-14, &error);
   //<-
   std::cout << std::setprecision(16) << Q << std::endl;
   std::cout << boost::math::relative_difference(Q, Q_expected) << std::endl;
   std::cout << fabs(Q - Q_expected) << std::endl;
   std::cout << error << std::endl;
   //->
   /*`
   This yields an actual error of zero, against an estimate of 4e-15.  In fact in this case the requested tolerance was almost
   certainly set too low: as we've seen above, for smooth functions, the precision achieved is often double
   that of the estimate, so if we integrate with a tolerance of 1e-9:
   */
   Q = gauss_kronrod<double, 15>::integrate(f1, 0, std::numeric_limits<double>::infinity(), 5, 1e-9, &error);
   //<-
   std::cout << std::setprecision(16) << Q << std::endl;
   std::cout << boost::math::relative_difference(Q, Q_expected) << std::endl;
   std::cout << fabs(Q - Q_expected) << std::endl;
   std::cout << error << std::endl;
   //->
   /*`
   We still achieve 1e-15 precision, with an error estimate of 1e-10.
   */
   //]
}

int main()
{
   gauss_examples();
   gauss_kronrod_examples();
   return 0;
}
