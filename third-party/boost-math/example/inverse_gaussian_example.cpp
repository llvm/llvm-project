// wald_example.cpp or inverse_gaussian_example.cpp

// Copyright Paul A. Bristow 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

// Example of using the Inverse Gaussian (or Inverse Normal) distribution.
// The Wald Distribution is


// Note that this file contains Quickbook mark-up as well as code
// and comments, don't change any of the special comment mark-ups!

//[inverse_gaussian_basic1
/*`
First we need some includes to access the normal distribution
(and some std output of course).
*/

#ifdef _MSC_VER
# pragma warning (disable : 4224)
# pragma warning (disable : 4189)
# pragma warning (disable : 4100)
# pragma warning (disable : 4224)
# pragma warning (disable : 4512)
# pragma warning (disable : 4702)
# pragma warning (disable : 4127)
#endif

//#define BOOST_MATH_INSTRUMENT

#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error
#define BOOST_MATH_DOMAIN_ERROR_POLICY ignore_error

#include <boost/math/distributions/inverse_gaussian.hpp> // for inverse_gaussian_distribution
  using boost::math::inverse_gaussian; // typedef provides default type is double.
  using boost::math::inverse_gaussian_distribution; // for inverse gaussian distribution.

#include <boost/math/distributions/normal.hpp> // for normal_distribution
using boost::math::normal; // typedef provides default type is double.

#include <boost/array.hpp>
using std::array;

#include <iostream>
  using std::cout; using std::endl; using std::left; using std::showpoint; using std::noshowpoint;
#include <iomanip>
  using std::setw; using std::setprecision;
#include <limits>
  using std::numeric_limits;
#include <sstream>
  using std::string;
#include <string>
  using std::stringstream;

// const double tol = 3 * numeric_limits<double>::epsilon();

int main()
{
  cout << "Example: Inverse Gaussian Distribution."<< endl;

 try
  {

      double tolfeweps = numeric_limits<double>::epsilon();
      //cout << "Tolerance = " << tol << endl;

      int precision = 17; // traditional tables are only computed to much lower precision.
      cout.precision(17); // std::numeric_limits<double>::max_digits10; for 64-bit doubles.

     // Traditional tables and values.
     double step = 0.2; // in z
      double range = 4; // min and max z = -range to +range.
      // Construct a (standard) inverse gaussian distribution s
      inverse_gaussian w11(1, 1);
      // (default mean = units, and standard deviation = unity)
      cout << "(Standard) Inverse Gaussian distribution, mean = "<< w11.mean()
          << ", scale = " << w11.scale() << endl;

/*` First the probability distribution function (pdf).
 */
      cout << "Probability distribution function (pdf) values" << endl;
      cout << "  z " "      pdf " << endl;
      cout.precision(5);
      for (double z = (numeric_limits<double>::min)(); z < range + step; z += step)
      {
        cout << left << setprecision(3) << setw(6) << z << " "
          << setprecision(precision) << setw(12) << pdf(w11, z) << endl;
      }
      cout.precision(6); // default
 /*`And the area under the normal curve from -[infin] up to z,
      the cumulative distribution function (cdf).
*/

      // For a (default) inverse gaussian distribution.
      cout << "Integral (area under the curve) from 0 up to z (cdf) " << endl;
      cout << "  z " "      cdf " << endl;
      for (double z = (numeric_limits<double>::min)(); z < range + step; z += step)
      {
        cout << left << setprecision(3) << setw(6) << z << " "
          << setprecision(precision) << setw(12) << cdf(w11, z) << endl;
      }
      /*`giving the following table:
[pre
    z       pdf
  2.23e-308 -1.#IND
  0.2    0.90052111680384117
  0.4    1.0055127039453111
  0.6    0.75123750098955733
  0.8    0.54377310461643302
  1      0.3989422804014327
  1.2    0.29846949816803292
  1.4    0.2274579835638664
  1.6    0.17614566625628389
  1.8    0.13829083543591469
  2      0.10984782236693062
  2.2    0.088133964251182237
  2.4    0.071327382959107177
  2.6    0.058162562161661699
  2.8    0.047742223328567722
  3      0.039418357969819712
  3.2    0.032715223861241892
  3.4    0.027278388940958308
  3.6    0.022840312999395804
  3.8    0.019196657941016954
  4      0.016189699458236451
  Integral (area under the curve) from 0 up to z (cdf)
    z       cdf
  2.23e-308 0
  0.2    0.063753567519976254
  0.4    0.2706136704424541
  0.6    0.44638391340412931
  0.8    0.57472390962590925
  1      0.66810200122317065
  1.2    0.73724578422952536
  1.4    0.78944214237790356
  1.6    0.82953458108474554
  1.8    0.86079282968276671
  2      0.88547542598600626
  2.2    0.90517870624273966
  2.4    0.92105495653509362
  2.6    0.93395164268166786
  2.8    0.94450240360053817
  3      0.95318792074278835
  3.2    0.96037753019309191
  3.4    0.96635823989417369
  3.6    0.97135533107998406
  3.8    0.97554722413538364
  4      0.97907636417888622
]

/*`We can get the inverse, the quantile, percentile, percentage point, or critical value
for a probability for a few probability from the above table, for z = 0.4, 1.0, 2.0:
*/
      cout << quantile(w11, 0.27061367044245421 ) << endl; // 0.4
      cout << quantile(w11, 0.66810200122317065) << endl; // 1.0
      cout << quantile(w11, 0.88547542598600615) << endl; // 2.0
/*`turning the expect values apart from some 'computational noise' in the least significant bit or two.

[pre
  0.40000000000000008
  0.99999999999999967
  1.9999999999999973
]

*/

    //  cout << "pnorm01(-0.406053) " << pnorm01(-0.406053) << ", cdfn01(-0.406053) = " << cdf(n01, -0.406053) << endl;
   //cout << "pnorm01(0.5) = " << pnorm01(0.5) << endl; // R pnorm(0.5,0,1) = 0.6914625  == 0.69146246127401312
    // R qnorm(0.6914625,0,1) = 0.5

    // formatC(SuppDists::qinvGauss(0.3649755481729598, 1, 1), digits=17)  [1] "0.50000000969034875"



  // formatC(SuppDists::dinvGauss(0.01, 1, 1), digits=17) [1] "2.0811768202028392e-19"
  // formatC(SuppDists::pinvGauss(0.01, 1, 1), digits=17) [1] "4.122313403318778e-23"



  //cout << " qinvgauss(0.3649755481729598, 1, 1) = " << qinvgauss(0.3649755481729598, 1, 1) << endl;  // 0.5
 // cout << quantile(s, 0.66810200122317065) << endl; // expect 1, get 0.50517388467190727
  //cout << " qinvgauss(0.62502320258649202, 1, 1) = " << qinvgauss(0.62502320258649202, 1, 1) << endl; // 0.9
  //cout << " qinvgauss(0.063753567519976254, 1, 1) = " << qinvgauss(0.063753567519976254, 1, 1) << endl; // 0.2
  //cout << " qinvgauss(0.0040761113207110162, 1, 1) = " << qinvgauss(0.0040761113207110162, 1, 1) << endl; // 0.1

  //double x = 1.; // SuppDists::pinvGauss(0.4, 1,1) [1] 0.2706137
  //double c = pinvgauss(x, 1, 1); // 0.3649755481729598 ==   cdf(x, 1,1) 0.36497554817295974
  //cout << "  pinvgauss(x, 1, 1) = " << c << endl; //  pinvgauss(x, 1, 1) = 0.27061367044245421
  //double p = pdf(w11, x);
  //double c = cdf(w11, x); // cdf(1, 1, 1) = 0.66810200122317065
  //cout << "cdf(" << x << ", " << w11.mean() << ", "<< w11.scale() << ") = " << c << endl; // cdf(x, 1, 1) 0.27061367044245421
  //cout << "pdf(" << x << ", " << w11.mean() << ", "<< w11.scale() << ") = " << p << endl;
  //double q = quantile(w11, c);
  //cout << "quantile(w11, " << c <<  ") = " << q << endl;

  //cout  << "quantile(w11, 4.122313403318778e-23) = "<< quantile(w11, 4.122313403318778e-23) << endl; // quantile
  //cout << "quantile(w11, 4.8791443010851493e-219) = " << quantile(w11, 4.8791443010851493e-219) << endl; // quantile

  //double c1 = 1 - cdf(w11, x); //  1 - cdf(1, 1, 1) = 0.33189799877682935
  //cout << "1 - cdf(" << x << ", " << w11.mean() << ", " << w11.scale() << ") = " << c1 << endl; // cdf(x, 1, 1) 0.27061367044245421
  //double cc = cdf(complement(w11, x));
  //cout << "cdf(complement(" << x << ", " << w11.mean() << ", "<< w11.scale() << ")) = " << cc << endl; // cdf(x, 1, 1) 0.27061367044245421
  //// 1 - cdf(1000, 1, 1) = 0
  //// cdf(complement(1000, 1, 1)) = 4.8694344366900402e-222

  //cout << "quantile(w11, " << c << ") = "<< quantile(w11, c) << endl; // quantile = 0.99999999999999978 == x = 1
  //cout << "quantile(w11, " << c << ") = "<< quantile(w11, 1 - c) << endl; // quantile complement. quantile(w11, 0.66810200122317065) = 0.46336593652340152
//  cout << "quantile(complement(w11, " << c << ")) = " << quantile(complement(w11, c)) << endl; // quantile complement                = 0.46336593652340163

  // cdf(1, 1, 1) = 0.66810200122317065
  // 1 - cdf(1, 1, 1) = 0.33189799877682935
  // cdf(complement(1, 1, 1)) = 0.33189799877682929

  // quantile(w11, 0.66810200122317065) = 0.99999999999999978
  // 1 - quantile(w11, 0.66810200122317065) = 2.2204460492503131e-016
  // quantile(complement(w11, 0.33189799877682929)) = 0.99999999999999989


  // qinvgauss(c, 1, 1) = 0.3999999999999998
  // SuppDists::qinvGauss(0.270613670442454, 1, 1) [1] 0.4


  /*
  double qs = pinvgaussU(c, 1, 1); //
    cout << "qinvgaussU(c, 1, 1) = " << qs << endl; // qinvgaussU(c, 1, 1) = 0.86567442459240929
    // > z=q - exp(c) * p [1] 0.8656744 qs 0.86567442459240929 double
    // Is this the complement?
    cout << "qgamma(0.2, 0.5, 1) expect 0.0320923 = " << qgamma(0.2, 0.5, 1) << endl;
    // qgamma(0.2, 0.5, 1) expect 0.0320923 = 0.032092377333650807


  cout << "qinvgauss(pinvgauss(x, 1, 1) = " << q
  << ", diff = " << x - q << ", fraction = " << (x - q) /x << endl; // 0.5

 */   // > SuppDists::pinvGauss(0.02, 1,1)  [1] 4.139176e-12
  // > SuppDists::qinvGauss(4.139176e-12, 1,1) [1] 0.02000000


    // pinvGauss(1,1,1) = 0.668102  C++  == 0.66810200122317065
  // qinvGauss(0.668102,1,1) = 1

   //  SuppDists::pinvGauss(0.3,1,1) = 0.1657266
  // cout << "qinvGauss(0.0040761113207110162, 1, 1) = " << qinvgauss(0.0040761113207110162, 1, 1) << endl;
  //cout << "quantile(s, 0.1657266) = " << quantile(s, 0.1657266) << endl; // expect 1.

  //wald s12(2, 1);
  //cout << "qinvGauss(0.3, 2, 1) = " << qinvgauss(0.3, 2, 1) << endl; // SuppDists::qinvGauss(0.3,2,1) == 0.58288065635052944
  //// but actually get qinvGauss(0.3, 2, 1) = 0.58288064777632187
  //cout  << "cdf(s12, 0.3) = " << cdf(s12, 0.3) << endl; //  cdf(s12, 0.3) = 0.10895339868447573

 // using boost::math::wald;
  //cout.precision(6);

 /*
 double m = 1;
  double l = 1;
  double x = 0.1;
  //c = cdf(w, x);
  double p = pinvgauss(x, m, l);
  cout << "x = " << x << ",  pinvgauss(x, m, l) = " << p << endl; // R 0.4 0.2706137
  double qg = qgamma(1.- p, 0.5, 1.0, true, false);
  cout << " qgamma(1.- p, 0.5, 1.0, true, false) = " << qg << endl; // 0.606817
  double g = guess_whitmore(p, m, l);
  cout << "m = " << m << ", l = " << l << ",   x = " << x << ", guess = " << g
    << ", diff = " << (x - g) << endl;

  g = guess_wheeler(p, m, l);
   cout << "m = " << m << ", l = " << l << ",   x = " << x << ", guess = " << g
    << ", diff = " << (x - g) << endl;

   g = guess_bagshaw(p, m, l);
   cout << "m = " << m << ", l = " << l << ",   x = " << x << ", guess = " << g
    << ", diff = " << (x - g) << endl;

   // m = 1, l = 10,   x = 0.9, guess = 0.89792, diff = 0.00231075 so a better fit.
  // x = 0.9, guess = 0.887907
  // x = 0.5, guess = 0.474977
  // x = 0.4, guess = 0.369597
  // x = 0.2, guess = 0.155196

  // m = 1, l = 2,   x = 0.9, guess = 1.0312, diff = -0.145778
  // m = 1, l = 2,   x = 0.1, guess = 0.122201, diff = -0.222013
  //  m = 1, l = 2,   x = 0.2, guess = 0.299326, diff = -0.49663
  //   m = 1, l = 2,   x = 0.5, guess = 1.00437, diff = -1.00875
  // m = 1, l = 2,   x = 0.7, guess = 1.01517, diff = -0.450247

  double ls[7] = {0.1, 0.2, 0.5, 1., 2., 10, 100}; // scale values.
  double ms[10] = {0.001, 0.02, 0.1, 0.2, 0.5, 0.9, 1., 2., 10, 100};  // mean values.
   */

    cout.precision(6); // Restore to default.
  } // try
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

inverse_gaussian_example.cpp
  inverse_gaussian_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Debug\inverse_gaussian_example.exe
  Example: Inverse Gaussian Distribution.
  (Standard) Inverse Gaussian distribution, mean = 1, scale = 1
  Probability distribution function (pdf) values
    z       pdf
  2.23e-308 -1.#IND
  0.2    0.90052111680384117
  0.4    1.0055127039453111
  0.6    0.75123750098955733
  0.8    0.54377310461643302
  1      0.3989422804014327
  1.2    0.29846949816803292
  1.4    0.2274579835638664
  1.6    0.17614566625628389
  1.8    0.13829083543591469
  2      0.10984782236693062
  2.2    0.088133964251182237
  2.4    0.071327382959107177
  2.6    0.058162562161661699
  2.8    0.047742223328567722
  3      0.039418357969819712
  3.2    0.032715223861241892
  3.4    0.027278388940958308
  3.6    0.022840312999395804
  3.8    0.019196657941016954
  4      0.016189699458236451
  Integral (area under the curve) from 0 up to z (cdf)
    z       cdf
  2.23e-308 0
  0.2    0.063753567519976254
  0.4    0.2706136704424541
  0.6    0.44638391340412931
  0.8    0.57472390962590925
  1      0.66810200122317065
  1.2    0.73724578422952536
  1.4    0.78944214237790356
  1.6    0.82953458108474554
  1.8    0.86079282968276671
  2      0.88547542598600626
  2.2    0.90517870624273966
  2.4    0.92105495653509362
  2.6    0.93395164268166786
  2.8    0.94450240360053817
  3      0.95318792074278835
  3.2    0.96037753019309191
  3.4    0.96635823989417369
  3.6    0.97135533107998406
  3.8    0.97554722413538364
  4      0.97907636417888622
  0.40000000000000008
  0.99999999999999967
  1.9999999999999973



> SuppDists::dinvGauss(2, 1, 1) [1] 0.1098478
> SuppDists::dinvGauss(0.4, 1, 1) [1] 1.005513
> SuppDists::dinvGauss(0.5, 1, 1) [1] 0.8787826
> SuppDists::dinvGauss(0.39, 1, 1) [1] 1.016559
> SuppDists::dinvGauss(0.38, 1, 1) [1] 1.027006
> SuppDists::dinvGauss(0.37, 1, 1) [1] 1.036748
> SuppDists::dinvGauss(0.36, 1, 1) [1] 1.045661
> SuppDists::dinvGauss(0.35, 1, 1) [1] 1.053611
> SuppDists::dinvGauss(0.3, 1, 1) [1] 1.072888
> SuppDists::dinvGauss(0.1, 1, 1) [1] 0.2197948
> SuppDists::dinvGauss(0.2, 1, 1) [1] 0.9005211
>
x = 0.3 [1, 1] 1.0728879234594337  // R SuppDists::dinvGauss(0.3, 1, 1) [1] 1.072888

x = 1   [1, 1] 0.3989422804014327


 0 "                NA"
 1 "0.3989422804014327"
 2 "0.10984782236693059"
 3 "0.039418357969819733"
 4 "0.016189699458236468"
 5 "0.007204168934430732"
 6 "0.003379893528659049"
 7 "0.0016462878258114036"
 8 "0.00082460931140859956"
 9 "0.00042207355643694234"
10 "0.00021979480031862676"


[1] "                NA"     " 0.690988298942671"     "0.11539974210409144"
 [4] "0.01799698883772935"    "0.0029555399206496469"  "0.00050863023587406587"
 [7] "9.0761842931362914e-05" "1.6655279133132795e-05" "3.1243174913715429e-06"
[10] "5.96530227727434e-07"   "1.1555606328299836e-07"


matC(dinvGauss(0:10, 1, 3), digits=17)  df = 3
[1] "                NA"     " 0.690988298942671"     "0.11539974210409144"
 [4] "0.01799698883772935"    "0.0029555399206496469"  "0.00050863023587406587"
 [7] "9.0761842931362914e-05" "1.6655279133132795e-05" "3.1243174913715429e-06"
[10] "5.96530227727434e-07"   "1.1555606328299836e-07"
$title
[1] "Inverse Gaussian"

$nu
[1] 1

$lambda
[1] 3

$Mean
[1] 1

$Median
[1] 0.8596309

$Mode
[1] 0.618034

$Variance
[1] 0.3333333

$SD
[1] 0.5773503

$ThirdCentralMoment
[1] 0.3333333

$FourthCentralMoment
[1] 0.8888889

$PearsonsSkewness...mean.minus.mode.div.SD
[1] 0.6615845

$Skewness...sqrtB1
[1] 1.732051

$Kurtosis...B2.minus.3
[1] 5

  Example: Wald distribution.
  (Standard) Wald distribution, mean = 1, scale = 1
  1 dx =      0.24890250442652451, x =      0.70924622051646713
  2 dx =    -0.038547954953794553, x =      0.46034371608994262
  3 dx =   -0.0011074090830291131, x =      0.49889167104373716
  4 dx = -9.1987259926368029e-007, x =      0.49999908012676625
  5 dx =  -6.346513344581067e-013, x =      0.49999999999936551
  dx = 6.3168242705156857e-017 at i = 6
   qinvgauss(0.3649755481729598, 1, 1) = 0.50000000000000011
  1 dx =       0.6719944578376621, x =       1.3735318786222666
  2 dx =     -0.16997432635769361, x =      0.70153742078460446
  3 dx =    -0.027865119206495724, x =      0.87151174714229807
  4 dx =  -0.00062283290009492603, x =      0.89937686634879377
  5 dx = -3.0075104108208687e-007, x =      0.89999969924888867
  6 dx = -7.0485322513588089e-014, x =      0.89999999999992975
  7 dx =   9.557331866250277e-016, x =      0.90000000000000024
  dx = 0 at i = 8
   qinvgauss(0.62502320258649202, 1, 1) = 0.89999999999999925
  1 dx =   -0.0052296256747447678, x =      0.19483508278446249
  2 dx =  6.4699046853900505e-005, x =      0.20006470845920726
  3 dx =  9.4123530465288137e-009, x =      0.20000000941235335
  4 dx =  2.7739513919147025e-016, x =      0.20000000000000032
  dx = 1.5410841066192808e-016 at i = 5
   qinvgauss(0.063753567519976254, 1, 1) = 0.20000000000000004
  1 dx =                       -1, x =     -0.46073286697416105
  2 dx =      0.47665501251497061, x =      0.53926713302583895
  3 dx =       -0.171105768635964, x =     0.062612120510868341
  4 dx =     0.091490360797512563, x =      0.23371788914683234
  5 dx =     0.029410311722649803, x =      0.14222752834931979
  6 dx =     0.010761845493592421, x =      0.11281721662666999
  7 dx =    0.0019864890597643035, x =      0.10205537113307757
  8 dx =  6.8800383732599561e-005, x =      0.10006888207331327
  9 dx =  8.1689466405590418e-008, x =      0.10000008168958067
  10 dx =   1.133634672475146e-013, x =      0.10000000000011428
  11 dx =  5.9588135045224526e-016, x =      0.10000000000000091
  12 dx =   3.433223674791152e-016, x =      0.10000000000000031
  dx = 9.0763384505974048e-017 at i = 13
   qinvgauss(0.0040761113207110162, 1, 1) = 0.099999999999999964


     wald_example.vcxproj -> J:\Cpp\MathToolkit\test\Math_test\Debug\wald_example.exe
  Example: Wald distribution.
  Tolerance = 6.66134e-016
  (Standard) Wald distribution, mean = 1, scale = 1
  cdf(x, 1,1) 4.1390252102096375e-012
  qinvgauss(pinvgauss(x, 1, 1) = 0.020116801973767886, diff = -0.00011680197376788548, fraction = -0.005840098688394274
  ____________________________________________________________
  wald 1, 1
  x =                     0.02, diff x - qinvgauss(cdf) = -0.00011680197376788548
  x =      0.10000000000000001, diff x - qinvgauss(cdf) = 8.7430063189231078e-016
  x =      0.20000000000000001, diff x - qinvgauss(cdf) = -1.1102230246251565e-016
  x =      0.29999999999999999, diff x - qinvgauss(cdf) = 0
  x =      0.40000000000000002, diff x - qinvgauss(cdf) = 2.2204460492503131e-016
  x =                      0.5, diff x - qinvgauss(cdf) = -1.1102230246251565e-016
  x =      0.59999999999999998, diff x - qinvgauss(cdf) = 1.1102230246251565e-016
  x =      0.80000000000000004, diff x - qinvgauss(cdf) = 1.1102230246251565e-016
  x =      0.90000000000000002, diff x - qinvgauss(cdf) = 0
  x =      0.98999999999999999, diff x - qinvgauss(cdf) = -1.1102230246251565e-016
  x =                    0.999, diff x - qinvgauss(cdf) = -1.1102230246251565e-016


*/



