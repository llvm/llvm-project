// inverse_chi_squared_distribution_find_df_example.cpp

// Copyright Paul A. Bristow 2010.
// Copyright Thomas Mang 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

//#define BOOST_MATH_INSTRUMENT

// Example 1 of using inverse chi squared distribution
#include <boost/math/distributions/inverse_chi_squared.hpp>
using boost::math::inverse_chi_squared_distribution;  // inverse_chi_squared_distribution.
using boost::math::inverse_chi_squared; //typedef for nverse_chi_squared_distribution double.

#include <iostream>
using std::cout;    using std::endl;
#include <iomanip> 
using std::setprecision;
using std::setw;
#include <cmath>
using std::sqrt;

int main()
{
  cout << "Example using Inverse chi squared distribution to find df. " << endl;
  try
  {
    cout.precision(std::numeric_limits<double>::max_digits10); // 
    int i = std::numeric_limits<double>::max_digits10;
    cout << "Show all potentially significant decimal digits std::numeric_limits<double>::max_digits10 = " << i << endl; 

    cout.precision(3);
    double nu = 10.;
    double scale1 = 1./ nu; // 1st definition sigma^2 = 1/df;
    double scale2 = 1.; // 2nd definition sigma^2 = 1
    inverse_chi_squared sichsq(nu, 1/nu); // Explicitly scaled to default scale = 1/df.
    inverse_chi_squared ichsq(nu); // Implicitly scaled to default scale = 1/df.
    // Try degrees of freedom estimator

    //double df = chi_squared::find_degrees_of_freedom(-diff, alpha[i], alpha[i], variance);

    cout << "ichsq.degrees_of_freedom() = " << ichsq.degrees_of_freedom() << endl;

    double diff = 0.5;  // difference from variance to detect (delta).
    double variance = 1.; // true variance
    double alpha = 0.9;
    double beta = 0.9;

    cout << "diff = " << diff 
      << ", variance = " << variance << ", ratio = " << diff/variance
      << ", alpha = " << alpha << ", beta = " << beta << endl;

    /* inverse_chi_square_df_estimator is not in the code base anymore?

    using boost::math::detail::inverse_chi_square_df_estimator;
    using boost::math::policies::default_policy;
    inverse_chi_square_df_estimator<> a_df(alpha, beta, variance, diff);

    cout << "df    est" << endl;
    for (double df = 1; df < 3; df += 0.1)
    {
      double est_df = a_df(1);
      cout << df << "    " << a_df(df) << endl;
    }
    */

    //template <class F, class T, class Tol, class Policy>std::pair<T, T> 
    // bracket_and_solve_root(F f, const T& guess, T factor, bool rising, Tol tol, std::uintmax_t& max_iter, const Policy& pol)


    // TODO: Not implemented
    //double df = inverse_chi_squared_distribution<>::find_degrees_of_freedom(diff, alpha, beta, variance, 0);
    //cout << df << endl;
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

  Example using Inverse chi squared distribution to find df. 
  Show all potentially significant decimal digits std::numeric_limits<double>::max_digits10 = 17
  10
  
  Message from thrown exception was:
     Error in function boost::math::inverse_chi_squared_distribution<double>::inverse_chi_squared_distribution: Degrees of freedom argument is 1.#INF, but must be > 0 !
diff = 0.5, variance = 1, ratio = 0.5, alpha = 0.1, beta = 0.1
  df    est
  1    1
  ratio+1 = 1.5, quantile(0.1) = 0.00618, cdf = 6.5e-037, result = -0.1
  1.1    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.00903, cdf = 1.2e-025, result = -0.1
  1.2    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0125, cdf = 8.25e-019, result = -0.1
  1.3    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0166, cdf = 2.17e-014, result = -0.1
  1.4    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0212, cdf = 2.2e-011, result = -0.1
  1.5    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0265, cdf = 3e-009, result = -0.1
  1.6    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0323, cdf = 1.11e-007, result = -0.1
  1.7    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0386, cdf = 1.7e-006, result = -0.1
  1.8    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0454, cdf = 1.41e-005, result = -0.1
  1.9    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0527, cdf = 7.55e-005, result = -0.1
  2    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0604, cdf = 0.000291, result = -0.1
  2.1    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0685, cdf = 0.00088, result = -0.1
  2.2    -0.1
  ratio+1 = 1.5, quantile(0.1) = 0.0771, cdf = 0.0022, result = -0.0999
  2.3    -0.0999
  ratio+1 = 1.5, quantile(0.1) = 0.0859, cdf = 0.00475, result = -0.0997
  2.4    -0.0997
  ratio+1 = 1.5, quantile(0.1) = 0.0952, cdf = 0.00911, result = -0.0993
  2.5    -0.0993
  ratio+1 = 1.5, quantile(0.1) = 0.105, cdf = 0.0159, result = -0.0984
  2.6    -0.0984
  ratio+1 = 1.5, quantile(0.1) = 0.115, cdf = 0.0257, result = -0.0967
  2.7    -0.0967
  ratio+1 = 1.5, quantile(0.1) = 0.125, cdf = 0.039, result = -0.094
  2.8    -0.094
  ratio+1 = 1.5, quantile(0.1) = 0.135, cdf = 0.056, result = -0.0897
  2.9    -0.0897
  ratio+1 = 1.5, quantile(0.1) = 20.6, cdf = 1, result = 0.9

    ichsq.degrees_of_freedom() = 10
  diff = 0.5, variance = 1, ratio = 0.5, alpha = 0.9, beta = 0.9
  df    est
  1    1
  ratio+1 = 1.5, quantile(0.9) = 0.729, cdf = 0.269, result = -0.729
  1.1    -0.729
  ratio+1 = 1.5, quantile(0.9) = 0.78, cdf = 0.314, result = -0.693
  1.2    -0.693
  ratio+1 = 1.5, quantile(0.9) = 0.83, cdf = 0.36, result = -0.655
  1.3    -0.655
  ratio+1 = 1.5, quantile(0.9) = 0.879, cdf = 0.405, result = -0.615
  1.4    -0.615
  ratio+1 = 1.5, quantile(0.9) = 0.926, cdf = 0.449, result = -0.575
  1.5    -0.575
  ratio+1 = 1.5, quantile(0.9) = 0.973, cdf = 0.492, result = -0.535
  1.6    -0.535
  ratio+1 = 1.5, quantile(0.9) = 1.02, cdf = 0.534, result = -0.495
  1.7    -0.495
  ratio+1 = 1.5, quantile(0.9) = 1.06, cdf = 0.574, result = -0.455
  1.8    -0.455
  ratio+1 = 1.5, quantile(0.9) = 1.11, cdf = 0.612, result = -0.417
  1.9    -0.417
  ratio+1 = 1.5, quantile(0.9) = 1.15, cdf = 0.648, result = -0.379
  2    -0.379
  ratio+1 = 1.5, quantile(0.9) = 1.19, cdf = 0.681, result = -0.342
  2.1    -0.342
  ratio+1 = 1.5, quantile(0.9) = 1.24, cdf = 0.713, result = -0.307
  2.2    -0.307
  ratio+1 = 1.5, quantile(0.9) = 1.28, cdf = 0.742, result = -0.274
  2.3    -0.274
  ratio+1 = 1.5, quantile(0.9) = 1.32, cdf = 0.769, result = -0.242
  2.4    -0.242
  ratio+1 = 1.5, quantile(0.9) = 1.36, cdf = 0.793, result = -0.212
  2.5    -0.212
  ratio+1 = 1.5, quantile(0.9) = 1.4, cdf = 0.816, result = -0.184
  2.6    -0.184
  ratio+1 = 1.5, quantile(0.9) = 1.44, cdf = 0.836, result = -0.157
  2.7    -0.157
  ratio+1 = 1.5, quantile(0.9) = 1.48, cdf = 0.855, result = -0.133
  2.8    -0.133
  ratio+1 = 1.5, quantile(0.9) = 1.52, cdf = 0.872, result = -0.11
  2.9    -0.11
  ratio+1 = 1.5, quantile(0.9) = 29.6, cdf = 1, result = 0.1


  */
  
