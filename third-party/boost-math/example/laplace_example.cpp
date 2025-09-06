// laplace_example.cpp

// Copyright Paul A. Bristow 2008, 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of using laplace (& comparing with normal) distribution.

// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[laplace_example1
/*`
First we need some includes to access the laplace & normal distributions
(and some std output of course).
*/

#include <boost/math/distributions/laplace.hpp> // for laplace_distribution
  using boost::math::laplace; // typedef provides default type is double.
#include <boost/math/distributions/normal.hpp> // for normal_distribution
  using boost::math::normal; // typedef provides default type is double.

#include <iostream>
  using std::cout; using std::endl; using std::left; using std::showpoint; using std::noshowpoint;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <limits>
  using std::numeric_limits;

int main()
{
  cout << "Example: Laplace distribution." << endl;

  try
  {
    { // Traditional tables and values.
/*`Let's start by printing some traditional tables.
*/      
      double step = 1.; // in z 
      double range = 4; // min and max z = -range to +range.
      //int precision = 17; // traditional tables are only computed to much lower precision.
      int precision = 4; // traditional table at much lower precision.
      int width = 10; // for use with setw.

      // Construct standard laplace & normal distributions l & s
        normal s; // (default location or mean = zero, and scale or standard deviation = unity)
        cout << "Standard normal distribution, mean or location = "<< s.location()
          << ", standard deviation or scale = " << s.scale() << endl;
        laplace l; // (default mean = zero, and standard deviation = unity)
        cout << "Laplace normal distribution, location = "<< l.location()
          << ", scale = " << l.scale() << endl;

/*` First the probability distribution function (pdf).
*/
      cout << "Probability distribution function values" << endl;
      cout << " z  PDF  normal     laplace    (difference)" << endl;
      cout.precision(5);
      for (double z = -range; z < range + step; z += step)
      {
        cout << left << setprecision(3) << setw(6) << z << " " 
          << setprecision(precision) << setw(width) << pdf(s, z) << "  "
          << setprecision(precision) << setw(width) << pdf(l, z)<<  "  ("
          << setprecision(precision) << setw(width) << pdf(l, z) - pdf(s, z) // difference.
          << ")" << endl;
      }
      cout.precision(6); // default
/*`Notice how the laplace is less at z = 1 , but has 'fatter' tails at 2 and 3. 

   And the area under the normal curve from -[infin] up to z,
   the cumulative distribution function (cdf).
*/
      // For a standard distribution 
      cout << "Standard location = "<< s.location()
        << ", scale = " << s.scale() << endl;
      cout << "Integral (area under the curve) from - infinity up to z " << endl;
      cout << " z  CDF  normal     laplace    (difference)" << endl;
      for (double z = -range; z < range + step; z += step)
      {
        cout << left << setprecision(3) << setw(6) << z << " " 
          << setprecision(precision) << setw(width) << cdf(s, z) << "  "
          << setprecision(precision) << setw(width) << cdf(l, z) <<  "  ("
          << setprecision(precision) << setw(width) << cdf(l, z) - cdf(s, z) // difference.
          << ")" << endl;
      }
      cout.precision(6); // default

/*`
Pretty-printing a traditional 2-dimensional table is left as an exercise for the student,
but why bother now that the Boost Math Toolkit lets you write
*/
    double z = 2.; 
    cout << "Area for gaussian z = " << z << " is " << cdf(s, z) << endl; // to get the area for z.
    cout << "Area for laplace z = " << z << " is " << cdf(l, z) << endl; // 
/*`
Correspondingly, we can obtain the traditional 'critical' values for significance levels.
For the 95% confidence level, the significance level usually called alpha,
is 0.05 = 1 - 0.95 (for a one-sided test), so we can write
*/
     cout << "95% of gaussian area has a z below " << quantile(s, 0.95) << endl;
     cout << "95% of laplace area has a z below " << quantile(l, 0.95) << endl;
   // 95% of area has a z below 1.64485
   // 95% of laplace area has a z below 2.30259
/*`and a two-sided test (a comparison between two levels, rather than a one-sided test)

*/
     cout << "95% of gaussian area has a z between " << quantile(s, 0.975)
       << " and " << -quantile(s, 0.975) << endl;
     cout << "95% of laplace area has a z between " << quantile(l, 0.975)
       << " and " << -quantile(l, 0.975) << endl;
   // 95% of area has a z between 1.95996 and -1.95996
   // 95% of laplace area has a z between 2.99573 and -2.99573
/*`Notice how much wider z has to be to enclose 95% of the area.
*/
  }
//] [/[laplace_example1]
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

Example: Laplace distribution.
Standard normal distribution, mean or location = 0, standard deviation or scale = 1
Laplace normal distribution, location = 0, scale = 1
Probability distribution function values
 z  PDF  normal     laplace    (difference)
-4     0.0001338   0.009158    (0.009024  )
-3     0.004432    0.02489     (0.02046   )
-2     0.05399     0.06767     (0.01368   )
-1     0.242       0.1839      (-0.05803  )
0      0.3989      0.5         (0.1011    )
1      0.242       0.1839      (-0.05803  )
2      0.05399     0.06767     (0.01368   )
3      0.004432    0.02489     (0.02046   )
4      0.0001338   0.009158    (0.009024  )
Standard location = 0, scale = 1
Integral (area under the curve) from - infinity up to z 
 z  CDF  normal     laplace    (difference)
-4     3.167e-005  0.009158    (0.009126  )
-3     0.00135     0.02489     (0.02354   )
-2     0.02275     0.06767     (0.04492   )
-1     0.1587      0.1839      (0.02528   )
0      0.5         0.5         (0         )
1      0.8413      0.8161      (-0.02528  )
2      0.9772      0.9323      (-0.04492  )
3      0.9987      0.9751      (-0.02354  )
4      1           0.9908      (-0.009126 )
Area for gaussian z = 2 is 0.97725
Area for laplace z = 2 is 0.932332
95% of gaussian area has a z below 1.64485
95% of laplace area has a z below 2.30259
95% of gaussian area has a z between 1.95996 and -1.95996
95% of laplace area has a z between 2.99573 and -2.99573

*/

