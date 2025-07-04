// Copyright Benjamin Sobotta 2012

//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#ifdef _MSC_VER
#  pragma warning (disable : 4512) // assignment operator could not be generated
#  pragma warning (disable : 4127) // conditional expression is constant.
#endif

#include <boost/math/distributions/skew_normal.hpp>
using boost::math::skew_normal_distribution;
using boost::math::skew_normal;
#include <iostream>
#include <cmath>
#include <utility>

void check(const double loc, const double sc, const double sh,
     const double * const cumulants, const std::pair<double, double> qu,
     const double x, const double tpdf, const double tcdf)
{
  using namespace boost::math;

  skew_normal D(loc, sc, sh);
  
  const double sk = cumulants[2] / (std::pow(cumulants[1], 1.5));
  const double kt = cumulants[3] / (cumulants[1] * cumulants[1]);

  // checks against tabulated values
  std::cout << "mean: table=" << cumulants[0] << "\tcompute=" << mean(D) << "\tdiff=" << fabs(cumulants[0]-mean(D)) << std::endl;
  std::cout << "var: table=" << cumulants[1] << "\tcompute=" << variance(D) << "\tdiff=" << fabs(cumulants[1]-variance(D)) << std::endl;
  std::cout << "skew: table=" << sk << "\tcompute=" << skewness(D) << "\tdiff=" << fabs(sk-skewness(D)) << std::endl;
  std::cout << "kur.: table=" << kt << "\tcompute=" << kurtosis_excess(D) << "\tdiff=" << fabs(kt-kurtosis_excess(D)) << std::endl;
  std::cout << "mode: table=" << "N/A" << "\tcompute=" << mode(D) << "\tdiff=" << "N/A" << std::endl;

  const double q = quantile(D, qu.first);
  const double cq = quantile(complement(D, qu.first));

  std::cout << "quantile(" << qu.first << "): table=" << qu.second << "\tcompute=" << q << "\tdiff=" << fabs(qu.second-q) << std::endl;

  // consistency
  std::cout << "cdf(quantile)=" << cdf(D, q) << "\tp=" << qu.first << "\tdiff=" << fabs(qu.first-cdf(D, q)) << std::endl;
  std::cout << "ccdf(cquantile)=" << cdf(complement(D,cq)) << "\tp=" << qu.first << "\tdiff=" << fabs(qu.first-cdf(complement(D,cq))) << std::endl;

  // PDF & CDF
  std::cout << "pdf(" << x << "): table=" << tpdf << "\tcompute=" << pdf(D,x) << "\tdiff=" << fabs(tpdf-pdf(D,x)) << std::endl;
  std::cout << "cdf(" << x << "): table=" << tcdf << "\tcompute=" << cdf(D,x) << "\tdiff=" << fabs(tcdf-cdf(D,x)) << std::endl;
  std::cout << "================================\n";
}

int main()
{
  using namespace boost::math;

  double sc = 0.0,loc,sh,x,dsn,qsn,psn,p;
  std::cout << std::setprecision(20);

  double cumulants[4];


  /* R:
     > install.packages("sn")
     Warning in install.packages("sn") :
     'lib = "/usr/lib64/R/library"' is not writable
     Would you like to create a personal library
     '~/R/x86_64-unknown-linux-gnu-library/2.12'
     to install packages into?  (y/n) y
     --- Please select a CRAN mirror for use in this session ---
     Loading Tcl/Tk interface ... done
     also installing the dependency mnormt

     trying URL 'http://mirrors.dotsrc.org/cran/src/contrib/mnormt_1.4-5.tar.gz'
     Content type 'application/x-gzip' length 34049 bytes (33 Kb)
     opened URL
     ==================================================
     downloaded 33 Kb

     trying URL 'http://mirrors.dotsrc.org/cran/src/contrib/sn_0.4-17.tar.gz'
     Content type 'application/x-gzip' length 65451 bytes (63 Kb)
     opened URL
     ==================================================
     downloaded 63 Kb


     > library(sn)
     > options(digits=22)


     > sn.cumulants(1.1, 2.2, -3.3)
     [1] -0.5799089925398568  2.0179057767837230 -2.0347951542374196
     [4]  2.2553488991015072
     > qsn(0.3, 1.1, 2.2, -3.3)
     [1] -1.180104068086876
     > psn(0.4, 1.1, 2.2, -3.3)
     [1] 0.733918618927874
     > dsn(0.4, 1.1, 2.2, -3.3)
     [1] 0.2941401101565995

   */

  //1 st
  loc = 1.1; sc = 2.2; sh = -3.3;
  std::cout << "location: " << loc << "\tscale: " << sc << "\tshape: " << sh << std::endl;
  cumulants[0] = -0.5799089925398568;
  cumulants[1] =  2.0179057767837230;
  cumulants[2] = -2.0347951542374196;
  cumulants[3] =  2.2553488991015072;
  x = 0.4;
  p = 0.3;
  qsn = -1.180104068086876;
  psn = 0.733918618927874;
  dsn = 0.2941401101565995;

  check(loc, sc, sh, cumulants, std::make_pair(p,qsn), x, dsn, psn);

  /* R:

     > sn.cumulants(1.1, .02, .03)
     [1] 1.1004785154529559e+00 3.9977102296128255e-04 4.7027439329779991e-11
     [4] 1.4847542790693825e-14
     > qsn(0.01, 1.1, .02, .03)
     [1] 1.053964962950150
     > psn(1.3, 1.1, .02, .03)
     [1] 1
     > dsn(1.3, 1.1, .02, .03)
     [1] 4.754580380601393e-21

   */

  // 2nd
  loc = 1.1; sc = .02; sh = .03;
  std::cout << "location: " << loc << "\tscale: " << sc << "\tshape: " << sh << std::endl;
  cumulants[0] = 1.1004785154529559e+00;
  cumulants[1] = 3.9977102296128255e-04;
  cumulants[2] = 4.7027439329779991e-11;
  cumulants[3] = 1.4847542790693825e-14;
  x = 1.3;
  p = 0.01;
  qsn = 1.053964962950150;
  psn = 1;
  dsn = 4.754580380601393e-21;

  check(loc, sc, sh, cumulants, std::make_pair(p,qsn), x, dsn, psn);

  /* R:

     > sn.cumulants(10.1, 5, -.03)
     [1]  9.980371136761052e+00  2.498568893508016e+01 -7.348037395278123e-04
     [4]  5.799821402614775e-05
     > qsn(.8, 10.1, 5, -.03)
     [1] 14.18727407491953
     > psn(-1.3, 10.1, 5, -.03)
     [1] 0.01201290665838824
     > dsn(-1.3, 10.1, 5, -.03)
     [1] 0.006254346348897927


  */

  // 3rd
  loc = 10.1; sc = 5; sh = -.03;
  std::cout << "location: " << loc << "\tscale: " << sc << "\tshape: " << sh << std::endl;
  cumulants[0] = 9.980371136761052e+00;
  cumulants[1] = 2.498568893508016e+01;
  cumulants[2] = -7.348037395278123e-04;
  cumulants[3] = 5.799821402614775e-05;
  x = -1.3;
  p = 0.8;
  qsn = 14.18727407491953;
  psn = 0.01201290665838824;
  dsn = 0.006254346348897927;

  check(loc, sc, sh, cumulants, std::make_pair(p,qsn), x, dsn, psn);


  /* R:

     > sn.cumulants(-10.1, 5, 30)
     [1] -6.112791696741384  9.102169946425548 27.206345266148194 71.572537838642916
     > qsn(.8, -10.1, 5, 30)
     [1] -3.692242172277
     > psn(-1.3, -10.1, 5, 30)
     [1] 0.921592193425035
     > dsn(-1.3, -10.1, 5, 30)
     [1] 0.0339105445232089

  */

  // 4th
  loc = -10.1; sc = 5; sh = 30;
  std::cout << "location: " << loc << "\tscale: " << sc << "\tshape: " << sh << std::endl;
  cumulants[0] = -6.112791696741384;
  cumulants[1] = 9.102169946425548;
  cumulants[2] = 27.206345266148194;
  cumulants[3] = 71.572537838642916;
  x = -1.3;
  p = 0.8;
  qsn = -3.692242172277;
  psn = 0.921592193425035;
  dsn = 0.0339105445232089;

  check(loc, sc, sh, cumulants, std::make_pair(p,qsn), x, dsn, psn);
  
  
  /* R:

     > sn.cumulants(0,1,5)
     [1] 0.7823901817554269 0.3878656034927102 0.2055576317962637 0.1061119471655128
     > qsn(0.5,0,1,5)
     [1] 0.674471117502844
     > psn(-0.5, 0,1,5)
     [1] 0.0002731513884140924
     > dsn(-0.5, 0,1,5)
     [1] 0.00437241570403263

  */

  // 5th
  loc = 0; sc = 1; sh = 5;
  std::cout << "location: " << loc << "\tscale: " << sc << "\tshape: " << sh << std::endl;
  cumulants[0] = 0.7823901817554269;
  cumulants[1] = 0.3878656034927102;
  cumulants[2] = 0.2055576317962637;
  cumulants[3] = 0.1061119471655128;
  x = -0.5;
  p = 0.5;
  qsn = 0.674471117502844;
  psn = 0.0002731513884140924;
  dsn = 0.00437241570403263;

  check(loc, sc, sh, cumulants, std::make_pair(p,qsn), x, dsn, psn);

  /* R:

     > sn.cumulants(0,1,1e5)
     [1] 0.7978845607629713 0.3633802276960805 0.2180136141122883 0.1147706820312645
     > qsn(0.5,0,1,1e5)
     [1] 0.6744897501960818
     > psn(-0.5, 0,1,1e5)
     [1] 0
     > dsn(-0.5, 0,1,1e5)
     [1] 0

  */

  // 6th
  loc = 0; sc = 1; sh = 1e5;
  std::cout << "location: " << loc << "\tscale: " << sc << "\tshape: " << sh << std::endl;
  cumulants[0] = 0.7978845607629713;
  cumulants[1] = 0.3633802276960805;
  cumulants[2] = 0.2180136141122883;
  cumulants[3] = 0.1147706820312645;
  x = -0.5;
  p = 0.5;
  qsn = 0.6744897501960818;
  psn = 0.;
  dsn = 0.;

  check(loc, sc, sh, cumulants, std::make_pair(p,qsn), x, dsn, psn);

  return 0;
}


// EOF








