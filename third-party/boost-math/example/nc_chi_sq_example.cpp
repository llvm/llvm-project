// Copyright John Maddock 2008
// Copyright Paul A. Bristow 2010, 2013
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Caution: this file contains Quickbook markup as well as code
// and comments, don't change any of the special comment markups!

//[nccs_eg

/*`

This example computes a table of the power of the [chi][super 2]
test at the 5% significance level, for various degrees of freedom
and non-centrality parameters.  The table is deliberately the same
as Table 6 from "The Non-Central [chi][super 2] and F-Distributions and
their applications.", P. B. Patnaik, Biometrika, Vol. 36, No. 1/2 (June 1949),
202-232.

First we need some includes to access the non-central chi squared distribution
(and some basic std output of course).

*/

#include <boost/math/distributions/non_central_chi_squared.hpp>
using boost::math::chi_squared;
using boost::math::non_central_chi_squared;

#include <iostream>
using std::cout; using std::endl;
using std::setprecision;

int main()
{
   /*`
   Create a table of the power of the [chi][super 2] test at
   5% significance level, start with a table header:
   */
   cout << "[table\n[[[nu]]";
   for(int lam = 2; lam <= 20; lam += 2)
   {
      cout << "[[lambda]=" << lam << "]";
   }
   cout << "]\n";

   /*`
   (Note: the enclosing [] brackets are to format as a table in Boost.Quickbook).

   Enumerate the rows and columns and print the power of the test
   for each table cell:
   */

   for(int n = 2; n <= 20; ++n)
   {
      cout << "[[" << n << "]";
      for(int lam = 2; lam <= 20; lam += 2)
      {
         /*`
         Calculate the [chi][super 2] statistic for a 5% significance:
         */
         double cs = quantile(complement(chi_squared(n), 0.05));
         /*`
         The power of the test is given by the complement of the CDF
         of the non-central [chi][super 2] distribution:
         */
         double beta = cdf(complement(non_central_chi_squared(n, lam), cs));
         /*`
         Then output the cell value:
         */
         cout << "[" << setprecision(3) << beta << "]";
      }
      cout << "]" << endl;
   }
   cout << "]" << endl;
}

/*`
The output from this program is a table in Boost.Quickbook format as shown below.

We can interpret this as follows - for example if [nu]=10 and [lambda]=10
then the power of the test is 0.542 - so we have only a 54% chance of
correctly detecting that our null hypothesis is false, and a 46% chance
of incurring a type II error (failing to reject the null hypothesis when
it is in fact false):

[table
[[[nu]][[lambda]=2][[lambda]=4][[lambda]=6][[lambda]=8][[lambda]=10][[lambda]=12][[lambda]=14][[lambda]=16][[lambda]=18][[lambda]=20]]
[[2][0.226][0.415][0.584][0.718][0.815][0.883][0.928][0.957][0.974][0.985]]
[[3][0.192][0.359][0.518][0.654][0.761][0.84][0.896][0.934][0.959][0.975]]
[[4][0.171][0.32][0.47][0.605][0.716][0.802][0.866][0.912][0.943][0.964]]
[[5][0.157][0.292][0.433][0.564][0.677][0.769][0.839][0.89][0.927][0.952]]
[[6][0.146][0.27][0.403][0.531][0.644][0.738][0.813][0.869][0.911][0.94]]
[[7][0.138][0.252][0.378][0.502][0.614][0.71][0.788][0.849][0.895][0.928]]
[[8][0.131][0.238][0.357][0.477][0.588][0.685][0.765][0.829][0.879][0.915]]
[[9][0.125][0.225][0.339][0.454][0.564][0.661][0.744][0.811][0.863][0.903]]
[[10][0.121][0.215][0.323][0.435][0.542][0.64][0.723][0.793][0.848][0.891]]
[[11][0.117][0.206][0.309][0.417][0.523][0.62][0.704][0.775][0.833][0.878]]
[[12][0.113][0.198][0.297][0.402][0.505][0.601][0.686][0.759][0.818][0.866]]
[[13][0.11][0.191][0.286][0.387][0.488][0.584][0.669][0.743][0.804][0.854]]
[[14][0.108][0.185][0.276][0.374][0.473][0.567][0.653][0.728][0.791][0.842]]
[[15][0.105][0.179][0.267][0.362][0.459][0.552][0.638][0.713][0.777][0.83]]
[[16][0.103][0.174][0.259][0.351][0.446][0.538][0.623][0.699][0.764][0.819]]
[[17][0.101][0.169][0.251][0.341][0.434][0.525][0.609][0.686][0.752][0.807]]
[[18][0.0992][0.165][0.244][0.332][0.423][0.512][0.596][0.673][0.74][0.796]]
[[19][0.0976][0.161][0.238][0.323][0.412][0.5][0.584][0.66][0.728][0.786]]
[[20][0.0961][0.158][0.232][0.315][0.402][0.489][0.572][0.648][0.716][0.775]]
]

See [@../../example/nc_chi_sq_example.cpp nc_chi_sq_example.cpp] for the full C++ source code.

*/

//]
