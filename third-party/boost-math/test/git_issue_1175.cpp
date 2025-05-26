//  (C) Copyright Matt Borland 2023.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include "math_unit_test.hpp"
#include <iostream>
#include <boost/math/distributions/beta.hpp>

using namespace std;
using boost::math::beta_distribution;

int main(int argc, char* argv[])
{
   double a = 5.0;
   double b = 5.0;
   double p = 0.5;

   beta_distribution<> dist(a, b);
   double x = quantile(dist, p);

   CHECK_ULP_CLOSE(x, 0.5, 2);

   return boost::math::test::report_errors();
}
