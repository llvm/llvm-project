//  (C) Copyright John Maddock 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

//
// As detailed in https://github.com/boostorg/math/issues/544
// our original non-central T test data wasn't accurate past about
// 20 decimal places.  This data generator takes the original input
// values and generates new CDF and CDF-complement values via
// tanh_sinh integration - an option that wasn't available when
// the original values were generated back in 2008.
//
// Note that this code will take SEVERAL DAYS to run on typical
// 2021 hardware.
//

#include <boost/math/special_functions/gamma.hpp>
#include <boost/math/special_functions/beta.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/constants/constants.hpp>
#include <boost/math/quadrature/exp_sinh.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/multiprecision/mpfr.hpp>
#include <fstream>

#include <table_type.hpp>

using namespace boost::math::tools;
using namespace boost::math;
using namespace std;
using namespace boost::multiprecision;

//using big_t = number<cpp_bin_float<200, digit_base_10, void, long long>>;
using big_t = number<mpfr_float_backend<200>>;

big_t nct_A(big_t v, big_t x, big_t nu)
{
   big_t result = boost::math::hypergeometric_1F1(v / 2 + 1, big_t(3) / 2, nu * nu * x * x / (2 * (v + x * x)));
   result *= boost::math::constants::root_two<big_t>() * nu * x;
   result /= (v + x * x) * boost::math::tgamma((v + 1) / 2);
   return result;
}
big_t nct_B(big_t v, big_t x, big_t nu)
{
   big_t result = boost::math::hypergeometric_1F1((v + 1) / 2, big_t(1) / 2, nu * nu * x * x / (2 * (x * x + v)));
   result /= sqrt(v + x * x) * boost::math::tgamma(v / 2 + 1);
   return result;
}
big_t nct_PDF(big_t v, big_t nu, big_t x)
{
   big_t result = nct_A(v, x, nu) + nct_B(v, x, nu);
   result *= pow(v, v / 2) * boost::math::tgamma(v + 1);
   result /= pow(big_t(2), v) * exp(nu * nu / 2) * pow(v + x * x, v / 2) * boost::math::tgamma(v / 2);
   return result;
}

big_t nc_t_F(big_t v, big_t nc, big_t t)
{
   unsigned j = 0;
   big_t sum = 0;
   big_t tol = std::numeric_limits<big_t>::epsilon();
   do
   {
      big_t x = t * t / (t * t + v);
      big_t p = exp(-nc * nc / 2) * pow(nc * nc / 2, j) / (2 * boost::math::factorial<big_t>(j));
      big_t q = (nc / 2) * exp(-nc * nc / 2) * pow(nc * nc / 2, j) / (boost::math::constants::root_two<big_t>() * boost::math::tgamma(j + big_t(3) / 2));
      big_t term = p * boost::math::ibeta(big_t(j) + 0.5, v / 2, x) + q * boost::math::ibeta(big_t(j + 1), v / 2, x);
      ++j;

      sum += term;

      if (fabs(sum * tol) > fabs(term))
         break;
   } while (true);

   return sum + boost::math::constants::half<big_t>() * (1 + boost::math::erf(-nc / boost::math::constants::root_two<big_t>()));
}
big_t nc_t(big_t v, big_t nc, big_t t)
{
   return t < 0 ? 1 - nc_t_F(v, -nc, t) : nc_t_F(v, nc, t);
}

#define SC_(x) BOOST_JOIN(x, f)

int main(int, char* [])
{
   mpfr_set_emax(mpfr_get_emax_max());
   mpfr_set_emin(mpfr_get_emin_min());

   boost::math::quadrature::exp_sinh<big_t> integrator(10);
   using T = float;

#include <nct.ipp>


   for (unsigned i = 0; i < nct.size(); ++i)
   {
      big_t error1, error2;
      big_t v(nct[i][0]), nc(nct[i][1]), x(nct[i][2]);
      big_t cdf, ccdf;
      try{
         cdf = integrator.integrate([&](big_t y) { return nct_PDF(v, nc, y); }, -std::numeric_limits<big_t>::infinity(), x, big_t(1e-36), &error1);
         ccdf = integrator.integrate([&](big_t y) { return nct_PDF(v, nc, y); }, x, std::numeric_limits<big_t>::infinity(), big_t(1e-36), &error2);
      }
      catch(const std::exception& e)
      {
         std::cout << "// " << e.what() << " reverting to ibeta method" << std::endl;
         error1 = error2 = 0;
         cdf = nc_t(v, nc, x);
         ccdf = 1 - cdf;
      }

      if (error1 > 1e-35)
      {
         std::cout << "// Accuracy for cdf was " << error1 << " reverting to ibeta method" << std::endl;
         cdf = nc_t(v, nc, x);
      }
      if (error2 > 1e-35)
      {
         std::cout << "// Accuracy for complement cdf was " << error2 << " reverting to ibeta method" << std::endl;
         ccdf = 1 - nc_t(v, nc, x);
      }

      std::cout << std::setprecision(40);
      std::cout << "{{ SC_(" << nct[i][0] << "), SC_(" << nct[i][1] << "), SC_(" << nct[i][2] << "), SC_(";
      std::cout << cdf << "), SC_(" << ccdf << ") }}," << std::endl;
   }

#include <nct_small_delta.ipp>
   for (unsigned i = 0; i < nct_small_delta.size(); ++i)
   {
      big_t error1, error2;
      big_t v(nct_small_delta[i][0]), nc(nct_small_delta[i][1]), x(nct_small_delta[i][2]);
      big_t cdf, ccdf;
      try {
         cdf = integrator.integrate([&](big_t y) { return nct_PDF(v, nc, y); }, -std::numeric_limits<big_t>::infinity(), x, big_t(1e-36), &error1);
         ccdf = integrator.integrate([&](big_t y) { return nct_PDF(v, nc, y); }, x, std::numeric_limits<big_t>::infinity(), big_t(1e-36), &error2);
      }
      catch (const std::exception& e)
      {
         std::cout << "// " << e.what() << " reverting to ibeta method" << std::endl;
         error1 = error2 = 0;
         cdf = nc_t(v, nc, x);
         ccdf = 1 - cdf;
      }

      if (error1 > 1e-35)
      {
         std::cout << "// Accuracy for cdf was " << error1 << " reverting to ibeta method" << std::endl;
         cdf = nc_t(v, nc, x);
      }
      if (error2 > 1e-35)
      {
         std::cout << "// Accuracy for complement cdf was " << error2 << " reverting to ibeta method" << std::endl;
         ccdf = 1 - nc_t(v, nc, x);
      }

      std::cout << std::setprecision(40);
      std::cout << "{{ SC_(" << v << "), SC_(" << nc << "), SC_(" << x << "), SC_(";
      std::cout << cdf << "), SC_(" << ccdf << ") }}," << std::endl;
   }

#include "../test/nct_asym.ipp"
   for (unsigned i = 0; i < nct_asym.size(); ++i)
   {
      big_t error1, error2;
      big_t v(nct_asym[i][0]), nc(nct_asym[i][1]), x(nct_asym[i][2]);
      big_t cdf, ccdf;
      try {
         cdf = integrator.integrate([&](big_t y) { return nct_PDF(v, nc, y); }, -std::numeric_limits<big_t>::infinity(), x, big_t(1e-36), &error1);
         ccdf = integrator.integrate([&](big_t y) { return nct_PDF(v, nc, y); }, x, std::numeric_limits<big_t>::infinity(), big_t(1e-36), &error2);
      }
      catch (const std::exception& e)
      {
         std::cout << "// " << e.what() << " reverting to ibeta method" << std::endl;
         error1 = error2 = 0;
         cdf = nc_t(v, nc, x);
         ccdf = 1 - cdf;
      }

      if (error1 > 1e-35)
      {
         std::cout << "// Accuracy for cdf was " << error1 << " reverting to ibeta method" << std::endl;
         cdf = nc_t(v, nc, x);
      }
      if (error2 > 1e-35)
      {
         std::cout << "// Accuracy for complement cdf was " << error2 << " reverting to ibeta method" << std::endl;
         ccdf = 1 - nc_t(v, nc, x);
      }

      std::cout << std::setprecision(40);
      std::cout << "{{ SC_(" << v << "), SC_(" << nc << "), SC_(" << x << "), SC_(";
      std::cout << cdf << "), SC_(" << ccdf << ") }}," << std::endl;
   }


   return 0;
}


