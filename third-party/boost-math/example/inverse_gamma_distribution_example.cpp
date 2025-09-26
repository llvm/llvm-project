// inverse_gamma_distribution_example.cpp

// Copyright Paul A. Bristow 2010.
// Copyright Thomas Mang 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example 1 of using inverse gamma
#include <boost/math/distributions/inverse_gamma.hpp>
using boost::math::inverse_gamma_distribution;  //  inverse_gamma_distribution.
using boost::math::inverse_gamma;

#include <boost/math/special_functions/gamma.hpp>
using boost::math::tgamma; // Used for naive pdf as a comparison.

#include <boost/math/distributions/gamma.hpp>
using boost::math::inverse_gamma_distribution;

#include <iostream>
using std::cout;    using std::endl;
#include <iomanip>
using std::setprecision;
#include <cmath>
using std::sqrt;

int main()
{

  cout << "Example using Inverse Gamma distribution. " << endl;
  // TODO - awaiting a real example using Bayesian statistics.

#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
  int max_digits10 = 2 + (boost::math::policies::digits<double, boost::math::policies::policy<> >() * 30103UL) / 100000UL;
  cout << "BOOST_NO_CXX11_NUMERIC_LIMITS is defined" << endl;
#else
  int max_digits10 = std::numeric_limits<double>::max_digits10;
#endif
  cout << "Show all potentially significant decimal digits std::numeric_limits<double>::max_digits10 = "
    << max_digits10 << endl;
  cout.precision(max_digits10); //

  double shape = 1.;
  double scale = 1.;
  double x = 0.5;
  // Construction using default RealType double, and default shape and scale..
  inverse_gamma_distribution<> my_inverse_gamma(shape, scale); // (alpha, beta)

  cout << "my_inverse_gamma.shape() = " << my_inverse_gamma.shape()
    << ", scale = "<< my_inverse_gamma.scale() << endl;
  cout << "x = " << x << ", pdf = " << pdf(my_inverse_gamma, x)
    << ", cdf = " << cdf(my_inverse_gamma, x) << endl;

  // Construct using  typedef and default shape and scale parameters.
  inverse_gamma my_ig;

  inverse_gamma my_ig23(2, 3);
  cout << "my_inverse_gamma.shape() = " << my_ig23.shape()
    << ", scale = "<< my_ig23.scale() << endl;
  cout << "x = " << x << ", pdf = " << pdf(my_ig23, x)
    << ", cdf = " << cdf(my_ig23, x) << endl;

  // Example of providing an 'out of domain' or 'bad' parameter,
  // here a shape < 1, for which mean is not defined.
  // Try block is essential to catch the exception message.
  // (Uses the default policy which is to throw on all errors).
  try
  {
    inverse_gamma if051(0.5, 1);
    //inverse_gamma if051(0.5, 1);
    cout << "mean(if051) = " << mean(if051) << endl;
  }
  catch(const std::exception& e)
  { // Always useful to include try & catch blocks because default policies
    // are to throw exceptions on arguments that cause errors like underflow, overflow.
    // Lacking try & catch blocks, the program will abort without a message below,
    // which may give some helpful clues as to the cause of the exception.
    std::cout <<
      "\n""Message from thrown exception was:\n   " << e.what() << std::endl;
  }

  return 0;
}  // int main()

/*

Output is:
  Example using Inverse Gamma distribution.
  std::numeric_limits<double>::max_digits10 = 17
  my_inverse_gamma.shape() = 1, scale = 1
  x = 0.5, pdf = 0.54134113294645081, cdf = 0.1353352832366127
  my_inverse_gamma.shape() = 2, scale = 3
  x = 0.5, pdf = 0.17847015671997774, cdf = 0.017351265236664509

  Message from thrown exception was:
     Error in function boost::math::mean(const inverse_gamma_distribution<double>&): Shape parameter is 0.5, but for a defined mean it must be > 1


*/


