// inverse_chi_squared_distribution_example.cpp

// Copyright Paul A. Bristow 2010.
// Copyright Thomas Mang 2010.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

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

template <class RealType>
RealType naive_pdf1(RealType df, RealType x)
{ // Formula from Wikipedia http://en.wikipedia.org/wiki/Inverse-chi-square_distribution
  // definition 1 using tgamma for simplicity as a check.
   using namespace std; // For ADL of std functions.
   using boost::math::tgamma;
   RealType df2 = df / 2;
   RealType result = (pow(2., -df2) * pow(x, (-df2 -1)) * exp(-1/(2 * x) ) )
      / tgamma(df2);  // 
   return result;
}

template <class RealType>
RealType naive_pdf2(RealType df, RealType x)
{ // Formula from Wikipedia http://en.wikipedia.org/wiki/Inverse-chi-square_distribution
  // Definition 2, using tgamma for simplicity as a check.
  // Not scaled, so assumes scale = 1 and only uses nu aka df.
   using namespace std; // For ADL of std functions.
   using boost::math::tgamma;
   RealType df2 = df / 2;
   RealType result = (pow(df2, df2) * pow(x, (-df2 -1)) * exp(-df/(2*x) ) )
     / tgamma(df2);
   return result;
}

template <class RealType>
RealType naive_pdf3(RealType df, RealType scale, RealType x)
{ // Formula from Wikipedia http://en.wikipedia.org/wiki/Scaled-inverse-chi-square_distribution
  // *Scaled* version, definition 3, df aka nu, scale aka sigma^2
  // using tgamma for simplicity as a check.
   using namespace std; // For ADL of std functions.
   using boost::math::tgamma;
   RealType df2 = df / 2;
   RealType result = (pow(scale * df2, df2) * exp(-df2 * scale/x) ) 
     / (tgamma(df2) * pow(x, 1+df2));
   return result;
}

template <class RealType>
RealType naive_pdf4(RealType df, RealType scale, RealType x)
{ // Formula from http://mathworld.wolfram.com/InverseChi-SquaredDistribution.html
  // Weisstein, Eric W. "Inverse Chi-Squared Distribution." From MathWorld--A Wolfram Web Resource.
  // *Scaled* version, definition 3, df aka nu, scale aka sigma^2
  // using tgamma for simplicity as a check.
   using namespace std; // For ADL of std functions.
   using boost::math::tgamma;
   RealType nu = df; // Wolfram uses greek symbols nu,
   RealType xi = scale; // and xi.
   RealType result = 
     pow(2, -nu/2) *  exp(- (nu * xi)/(2 * x)) * pow(x, -1-nu/2) * pow((nu * xi), nu/2) 
     / tgamma(nu/2);
   return result;
}

int main()
{

  cout << "Example (basic) using Inverse chi squared distribution. " << endl;

  // TODO find a more practical/useful example.  Suggestions welcome?

#ifdef BOOST_NO_CXX11_NUMERIC_LIMITS
  int max_digits10 = 2 + (boost::math::policies::digits<double, boost::math::policies::policy<> >() * 30103UL) / 100000UL;
  cout << "BOOST_NO_CXX11_NUMERIC_LIMITS is defined" << endl; 
#else 
  int max_digits10 = std::numeric_limits<double>::max_digits10;
#endif
  cout << "Show all potentially significant decimal digits std::numeric_limits<double>::max_digits10 = "
    << max_digits10 << endl; 
  cout.precision(max_digits10); // 

  inverse_chi_squared ichsqdef; // All defaults  - not very useful!
  cout << "default df = " << ichsqdef.degrees_of_freedom()
    << ", default scale = " <<  ichsqdef.scale() << endl; //  default df = 1, default scale = 0.5

   inverse_chi_squared ichsqdef4(4); // Unscaled version, default scale = 1 / degrees_of_freedom
   cout << "default df = " << ichsqdef4.degrees_of_freedom()
    << ", default scale = " <<  ichsqdef4.scale() << endl; //  default df = 4, default scale = 2

   inverse_chi_squared ichsqdef32(3, 2); // Scaled version, both degrees_of_freedom and scale specified.
   cout << "default df = " << ichsqdef32.degrees_of_freedom()
    << ", default scale = " <<  ichsqdef32.scale() << endl; // default df = 3, default scale = 2

  {
    cout.precision(3);
    double nu = 5.;
    //double scale1 = 1./ nu; // 1st definition sigma^2 = 1/df;
    //double scale2 = 1.; // 2nd definition sigma^2 = 1
    inverse_chi_squared ichsq(nu, 1/nu); // Not scaled
    inverse_chi_squared sichsq(nu, 1/nu); // scaled

    cout << "nu = " << ichsq.degrees_of_freedom() << ", scale = " << ichsq.scale() << endl;

    int width = 8;

    cout << "     x        pdf      pdf1   pdf2  pdf(scaled)    pdf       pdf      cdf     cdf" << endl;
    for (double x = 0.0; x < 1.; x += 0.1)
    {
      cout 
        << setw(width) << x 
        << ' ' << setw(width) << pdf(ichsq, x) // unscaled
        << ' ' << setw(width) << naive_pdf1(nu,  x) // Wiki def 1 unscaled matches graph 
        << ' ' << setw(width) << naive_pdf2(nu,  x) // scale = 1 - 2nd definition.
        << ' ' << setw(width) << naive_pdf3(nu, 1/nu, x) // scaled 
        << ' ' << setw(width) << naive_pdf4(nu, 1/nu, x) // scaled 
        << ' ' << setw(width) << pdf(sichsq, x)  // scaled
        << ' ' << setw(width) << cdf(sichsq, x)  // scaled
        << ' ' << setw(width) << cdf(ichsq, x)  // unscaled
       << endl;
    }
  }

  cout.precision(max_digits10);

  inverse_chi_squared ichisq(2., 0.5);
  cout << "pdf(ichisq, 1.) = " << pdf(ichisq, 1.) << endl;
  cout << "cdf(ichisq, 1.) = " << cdf(ichisq, 1.) << endl;

  return 0;
}  // int main()

/*

Output is:
 Example (basic) using Inverse chi squared distribution. 
  Show all potentially significant decimal digits std::numeric_limits<double>::max_digits10 = 17
  default df = 1, default scale = 1
  default df = 4, default scale = 0.25
  default df = 3, default scale = 2
  nu = 5, scale = 0.2
       x        pdf      pdf1   pdf2  pdf(scaled)    pdf       pdf      cdf     cdf
         0        0    -1.#J    -1.#J    -1.#J    -1.#J        0        0        0
       0.1     2.83     2.83 3.26e-007     2.83     2.83     2.83   0.0752   0.0752
       0.2     3.05     3.05  0.00774     3.05     3.05     3.05    0.416    0.416
       0.3      1.7      1.7    0.121      1.7      1.7      1.7    0.649    0.649
       0.4    0.941    0.941    0.355    0.941    0.941    0.941    0.776    0.776
       0.5    0.553    0.553    0.567    0.553    0.553    0.553    0.849    0.849
       0.6    0.345    0.345    0.689    0.345    0.345    0.345    0.893    0.893
       0.7    0.227    0.227    0.728    0.227    0.227    0.227    0.921    0.921
       0.8    0.155    0.155    0.713    0.155    0.155    0.155     0.94     0.94
       0.9     0.11     0.11    0.668     0.11     0.11     0.11    0.953    0.953
         1   0.0807   0.0807     0.61   0.0807   0.0807   0.0807    0.963    0.963
  pdf(ichisq, 1.) = 0.30326532985631671
  cdf(ichisq, 1.) = 0.60653065971263365


*/

