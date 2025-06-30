// Copyright John Maddock 2006.
// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#define NOMINMAX

#include "mp_t.hpp"
#include <boost/math/tools/test_data.hpp>
#include <boost/test/included/prg_exec_monitor.hpp>
#include <boost/math/special_functions/ellint_rj.hpp>
#include <boost/math/special_functions/ellint_rd.hpp>
#include <fstream>
#include <boost/math/tools/test_data.hpp>
#include <boost/random.hpp>

float extern_val;
// confuse the compilers optimiser, and force a truncation to float precision:
float truncate_to_float(float const * pf)
{
   extern_val = *pf;
   return *pf;
}

//
// Archived here is the original implementation of this
// function by Xiaogang Zhang, we can use this to
// generate special test cases for the new version:
//
template <typename T, typename Policy>
T ellint_rj_old(T x, T y, T z, T p, const Policy& pol)
{
   T value, u, lambda, alpha, beta, sigma, factor, tolerance;
   T X, Y, Z, P, EA, EB, EC, E2, E3, S1, S2, S3;
   unsigned long k;

   BOOST_MATH_STD_USING
      using namespace boost::math;

   static const char* function = "boost::math::ellint_rj<%1%>(%1%,%1%,%1%)";

   if(x < 0)
   {
      return policies::raise_domain_error<T>(function,
         "Argument x must be non-negative, but got x = %1%", x, pol);
   }
   if(y < 0)
   {
      return policies::raise_domain_error<T>(function,
         "Argument y must be non-negative, but got y = %1%", y, pol);
   }
   if(z < 0)
   {
      return policies::raise_domain_error<T>(function,
         "Argument z must be non-negative, but got z = %1%", z, pol);
   }
   if(p == 0)
   {
      return policies::raise_domain_error<T>(function,
         "Argument p must not be zero, but got p = %1%", p, pol);
   }
   if(x + y == 0 || y + z == 0 || z + x == 0)
   {
      return policies::raise_domain_error<T>(function,
         "At most one argument can be zero, "
         "only possible result is %1%.", std::numeric_limits<T>::quiet_NaN(), pol);
   }

   // error scales as the 6th power of tolerance
   tolerance = pow(T(1) * tools::epsilon<T>() / 3, T(1) / 6);

   // for p < 0, the integral is singular, return Cauchy principal value
   if(p < 0)
   {
      //
      // We must ensure that (z - y) * (y - x) is positive.
      // Since the integral is symmetrical in x, y and z
      // we can just permute the values:
      //
      if(x > y)
         std::swap(x, y);
      if(y > z)
         std::swap(y, z);
      if(x > y)
         std::swap(x, y);

      T q = -p;
      T pmy = (z - y) * (y - x) / (y + q);  // p - y

      BOOST_MATH_ASSERT(pmy >= 0);

      p = pmy + y;
      value = ellint_rj_old(x, y, z, p, pol);
      value *= pmy;
      value -= 3 * boost::math::ellint_rf(x, y, z, pol);
      value += 3 * sqrt((x * y * z) / (x * z + p * q)) * boost::math::ellint_rc(x * z + p * q, p * q, pol);
      value /= (y + q);
      return value;
   }

   // duplication
   sigma = 0;
   factor = 1;
   k = 1;
   do
   {
      u = (x + y + z + p + p) / 5;
      X = (u - x) / u;
      Y = (u - y) / u;
      Z = (u - z) / u;
      P = (u - p) / u;

      if((tools::max)(abs(X), abs(Y), abs(Z), abs(P)) < tolerance)
         break;

      T sx = sqrt(x);
      T sy = sqrt(y);
      T sz = sqrt(z);

      lambda = sy * (sx + sz) + sz * sx;
      alpha = p * (sx + sy + sz) + sx * sy * sz;
      alpha *= alpha;
      beta = p * (p + lambda) * (p + lambda);
      sigma += factor * boost::math::ellint_rc(alpha, beta, pol);
      factor /= 4;
      x = (x + lambda) / 4;
      y = (y + lambda) / 4;
      z = (z + lambda) / 4;
      p = (p + lambda) / 4;
      ++k;
   } while(k < policies::get_max_series_iterations<Policy>());

   // Check to see if we gave up too soon:
   policies::check_series_iterations<T>(function, k, pol);

   // Taylor series expansion to the 5th order
   EA = X * Y + Y * Z + Z * X;
   EB = X * Y * Z;
   EC = P * P;
   E2 = EA - 3 * EC;
   E3 = EB + 2 * P * (EA - EC);
   S1 = 1 + E2 * (E2 * T(9) / 88 - E3 * T(9) / 52 - T(3) / 14);
   S2 = EB * (T(1) / 6 + P * (T(-6) / 22 + P * T(3) / 26));
   S3 = P * ((EA - EC) / 3 - P * EA * T(3) / 22);
   value = 3 * sigma + factor * (S1 + S2 + S3) / (u * sqrt(u));

   return value;
}

template <typename T, typename Policy>
T ellint_rd_imp_old(T x, T y, T z, const Policy& pol)
{
   T value, u, lambda, sigma, factor, tolerance;
   T X, Y, Z, EA, EB, EC, ED, EE, S1, S2;
   unsigned long k;

   BOOST_MATH_STD_USING
   using namespace boost::math;

   static const char* function = "boost::math::ellint_rd<%1%>(%1%,%1%,%1%)";

   if(x < 0)
   {
      return policies::raise_domain_error<T>(function,
         "Argument x must be >= 0, but got %1%", x, pol);
   }
   if(y < 0)
   {
      return policies::raise_domain_error<T>(function,
         "Argument y must be >= 0, but got %1%", y, pol);
   }
   if(z <= 0)
   {
      return policies::raise_domain_error<T>(function,
         "Argument z must be > 0, but got %1%", z, pol);
   }
   if(x + y == 0)
   {
      return policies::raise_domain_error<T>(function,
         "At most one argument can be zero, but got, x + y = %1%", x + y, pol);
   }

   // error scales as the 6th power of tolerance
   tolerance = pow(tools::epsilon<T>() / 3, T(1) / 6);

   // duplication
   sigma = 0;
   factor = 1;
   k = 1;
   do
   {
      u = (x + y + z + z + z) / 5;
      X = (u - x) / u;
      Y = (u - y) / u;
      Z = (u - z) / u;
      if((tools::max)(abs(X), abs(Y), abs(Z)) < tolerance)
         break;
      T sx = sqrt(x);
      T sy = sqrt(y);
      T sz = sqrt(z);
      lambda = sy * (sx + sz) + sz * sx; //sqrt(x * y) + sqrt(y * z) + sqrt(z * x);
      sigma += factor / (sz * (z + lambda));
      factor /= 4;
      x = (x + lambda) / 4;
      y = (y + lambda) / 4;
      z = (z + lambda) / 4;
      ++k;
   } while(k < policies::get_max_series_iterations<Policy>());

   // Check to see if we gave up too soon:
   policies::check_series_iterations<T>(function, k, pol);

   // Taylor series expansion to the 5th order
   EA = X * Y;
   EB = Z * Z;
   EC = EA - EB;
   ED = EA - 6 * EB;
   EE = ED + EC + EC;
   S1 = ED * (ED * T(9) / 88 - Z * EE * T(9) / 52 - T(3) / 14);
   S2 = Z * (EE / 6 + Z * (-EC * T(9) / 22 + Z * EA * T(3) / 26));
   value = 3 * sigma + factor * (1 + S1 + S2) / (u * sqrt(u));

   return value;
}

template <typename T, typename Policy>
T ellint_rf_imp_old(T x, T y, T z, const Policy& pol)
{
   T value, X, Y, Z, E2, E3, u, lambda, tolerance;
   unsigned long k;
   BOOST_MATH_STD_USING
   using namespace boost::math;
   static const char* function = "boost::math::ellint_rf<%1%>(%1%,%1%,%1%)";
   if(x < 0 || y < 0 || z < 0)
   {
      return policies::raise_domain_error<T>(function,
         "domain error, all arguments must be non-negative, "
         "only sensible result is %1%.",
         std::numeric_limits<T>::quiet_NaN(), pol);
   }
   if(x + y == 0 || y + z == 0 || z + x == 0)
   {
      return policies::raise_domain_error<T>(function,
         "domain error, at most one argument can be zero, "
         "only sensible result is %1%.",
         std::numeric_limits<T>::quiet_NaN(), pol);
   }
   // Carlson scales error as the 6th power of tolerance,
   // but this seems not to work for types larger than
   // 80-bit reals, this heuristic seems to work OK:
   if(policies::digits<T, Policy>() > 64)
   {
      tolerance = pow(tools::epsilon<T>(), T(1) / 4.25f);
      BOOST_MATH_INSTRUMENT_VARIABLE(tolerance);
   }
   else
   {
      tolerance = pow(4 * tools::epsilon<T>(), T(1) / 6);
      BOOST_MATH_INSTRUMENT_VARIABLE(tolerance);
   }
   // duplication
   k = 1;
   do
   {
      u = (x + y + z) / 3;
      X = (u - x) / u;
      Y = (u - y) / u;
      Z = (u - z) / u;
      // Termination condition:
      if((tools::max)(abs(X), abs(Y), abs(Z)) < tolerance)
         break;
      T sx = sqrt(x);
      T sy = sqrt(y);
      T sz = sqrt(z);
      lambda = sy * (sx + sz) + sz * sx;
      x = (x + lambda) / 4;
      y = (y + lambda) / 4;
      z = (z + lambda) / 4;
      ++k;
   } while(k < policies::get_max_series_iterations<Policy>());
   // Check to see if we gave up too soon:
   policies::check_series_iterations<T>(function, k, pol);
   BOOST_MATH_INSTRUMENT_VARIABLE(k);
   // Taylor series expansion to the 5th order
   E2 = X * Y - Z * Z;
   E3 = X * Y * Z;
   value = (1 + E2*(E2 / 24 - E3*T(3) / 44 - T(0.1)) + E3 / 14) / sqrt(u);
   BOOST_MATH_INSTRUMENT_VARIABLE(value);
   return value;
}



boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rj_data_4e(mp_t n)
{
   mp_t result = ellint_rj_old(n, n, n, n, boost::math::policies::policy<>());
   return boost::math::make_tuple(n, n, n, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t> generate_rj_data_3e(mp_t x, mp_t p)
{
   mp_t r = ellint_rj_old(x, x, x, p, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, x, x, p, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t> generate_rj_data_2e_1(mp_t x, mp_t y, mp_t p)
{
   mp_t r = ellint_rj_old(x, x, y, p, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, x, y, p, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t> generate_rj_data_2e_2(mp_t x, mp_t y, mp_t p)
{
   mp_t r = ellint_rj_old(x, y, x, p, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, y, x, p, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t> generate_rj_data_2e_3(mp_t x, mp_t y, mp_t p)
{
   mp_t r = ellint_rj_old(y, x, x, p, boost::math::policies::policy<>());
   return boost::math::make_tuple(y, x, x, p, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t> generate_rj_data_2e_4(mp_t x, mp_t y, mp_t p)
{
   mp_t r = ellint_rj_old(x, y, p, p, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, y, p, p, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rd_data_2e_1(mp_t x, mp_t y)
{
   mp_t r = ellint_rd_imp_old(x, y, y, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, y, y, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rd_data_2e_2(mp_t x, mp_t y)
{
   mp_t r = ellint_rd_imp_old(x, x, y, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, x, y, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rd_data_2e_3(mp_t x)
{
   mp_t r = ellint_rd_imp_old(mp_t(0), x, x, boost::math::policies::policy<>());
   return boost::math::make_tuple(0, x, x, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rd_data_3e(mp_t x)
{
   mp_t r = ellint_rd_imp_old(x, x, x, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, x, x, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rd_data_0xy(mp_t x, mp_t y)
{
   mp_t r = ellint_rd_imp_old(mp_t(0), x, y, boost::math::policies::policy<>());
   return boost::math::make_tuple(mp_t(0), x, y, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rf_data_xxx(mp_t x)
{
   mp_t r = ellint_rf_imp_old(x, x, x, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, x, x, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rf_data_xyy(mp_t x, mp_t y)
{
   mp_t r = ellint_rf_imp_old(x, y, y, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, y, y, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rf_data_xxy(mp_t x, mp_t y)
{
   mp_t r = ellint_rf_imp_old(x, x, y, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, x, y, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rf_data_xyx(mp_t x, mp_t y)
{
   mp_t r = ellint_rf_imp_old(x, y, x, boost::math::policies::policy<>());
   return boost::math::make_tuple(x, y, x, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rf_data_0yy(mp_t y)
{
   mp_t r = ellint_rf_imp_old(mp_t(0), y, y, boost::math::policies::policy<>());
   return boost::math::make_tuple(mp_t(0), y, y, r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rf_data_xy0(mp_t x, mp_t y)
{
   mp_t r = ellint_rf_imp_old(x, y, mp_t(0), boost::math::policies::policy<>());
   return boost::math::make_tuple(x, y, mp_t(0), r);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rf_data(mp_t n)
{
   static boost::mt19937 r;
   boost::uniform_real<float> ur(0, 1);
   boost::uniform_int<int> ui(-100, 100);
   float x = ur(r);
   x = ldexp(x, ui(r));
   mp_t xr(truncate_to_float(&x));
   float y = ur(r);
   y = ldexp(y, ui(r));
   mp_t yr(truncate_to_float(&y));
   float z = ur(r);
   z = ldexp(z, ui(r));
   mp_t zr(truncate_to_float(&z));

   mp_t result = boost::math::ellint_rf(xr, yr, zr);
   return boost::math::make_tuple(xr, yr, zr, result);
}

boost::math::tuple<mp_t, mp_t, mp_t> generate_rc_data(mp_t n)
{
   static boost::mt19937 r;
   boost::uniform_real<float> ur(0, 1);
   boost::uniform_int<int> ui(-100, 100);
   float x = ur(r);
   x = ldexp(x, ui(r));
   mp_t xr(truncate_to_float(&x));
   float y = ur(r);
   y = ldexp(y, ui(r));
   mp_t yr(truncate_to_float(&y));

   mp_t result = boost::math::ellint_rc(xr, yr);
   return boost::math::make_tuple(xr, yr, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t, mp_t> generate_rj_data(mp_t n)
{
   static boost::mt19937 r;
   boost::uniform_real<float> ur(0, 1);
   boost::uniform_real<float> nur(-1, 1);
   boost::uniform_int<int> ui(-100, 100);
   float x = ur(r);
   x = ldexp(x, ui(r));
   mp_t xr(truncate_to_float(&x));
   float y = ur(r);
   y = ldexp(y, ui(r));
   mp_t yr(truncate_to_float(&y));
   float z = ur(r);
   z = ldexp(z, ui(r));
   mp_t zr(truncate_to_float(&z));
   float p = nur(r);
   p = ldexp(p, ui(r));
   mp_t pr(truncate_to_float(&p));

   boost::math::ellint_rj(x, y, z, p);

   mp_t result = boost::math::ellint_rj(xr, yr, zr, pr);
   return boost::math::make_tuple(xr, yr, zr, pr, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rd_data(mp_t n)
{
   static boost::mt19937 r;
   boost::uniform_real<float> ur(0, 1);
   boost::uniform_int<int> ui(-100, 100);
   float x = ur(r);
   x = ldexp(x, ui(r));
   mp_t xr(truncate_to_float(&x));
   float y = ur(r);
   y = ldexp(y, ui(r));
   mp_t yr(truncate_to_float(&y));
   float z = ur(r);
   z = ldexp(z, ui(r));
   mp_t zr(truncate_to_float(&z));

   mp_t result = boost::math::ellint_rd(xr, yr, zr);
   return boost::math::make_tuple(xr, yr, zr, result);
}

mp_t rg_imp(mp_t x, mp_t y, mp_t z)
{
   using std::swap;
   // If z is zero permute so the call to RD is valid:
   if(z == 0)
      swap(x, z);
   return (z * ellint_rf_imp_old(x, y, z, boost::math::policies::policy<>())
      - (x - z) * (y - z) * ellint_rd_imp_old(x, y, z, boost::math::policies::policy<>()) / 3
      + sqrt(x * y / z)) / 2;
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_data(mp_t n)
{
   static boost::mt19937 r;
   boost::uniform_real<float> ur(0, 1);
   boost::uniform_int<int> ui(-100, 100);
   float x = ur(r);
   x = ldexp(x, ui(r));
   mp_t xr(truncate_to_float(&x));
   float y = ur(r);
   y = ldexp(y, ui(r));
   mp_t yr(truncate_to_float(&y));
   float z = ur(r);
   z = ldexp(z, ui(r));
   mp_t zr(truncate_to_float(&z));

   mp_t result = rg_imp(xr, yr, zr);
   return boost::math::make_tuple(xr, yr, zr, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_xxx(mp_t x)
{
   mp_t result = rg_imp(x, x, x);
   return boost::math::make_tuple(x, x, x, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_xyy(mp_t x, mp_t y)
{
   mp_t result = rg_imp(x, y, y);
   return boost::math::make_tuple(x, y, y, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_xxy(mp_t x, mp_t y)
{
   mp_t result = rg_imp(x, x, y);
   return boost::math::make_tuple(x, x, y, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_xyx(mp_t x, mp_t y)
{
   mp_t result = rg_imp(x, y, x);
   return boost::math::make_tuple(x, y, x, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_0xx(mp_t x)
{
   mp_t result = rg_imp(mp_t(0), x, x);
   return boost::math::make_tuple(mp_t(0), x, x, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_x0x(mp_t x)
{
   mp_t result = rg_imp(x, mp_t(0), x);
   return boost::math::make_tuple(x, mp_t(0), x, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_xx0(mp_t x)
{
   mp_t result = rg_imp(x, x, mp_t(0));
   return boost::math::make_tuple(x, x, mp_t(0), result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_00x(mp_t x)
{
   mp_t result = sqrt(x) / 2;
   return boost::math::make_tuple(mp_t(0), mp_t(0), x, result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_0x0(mp_t x)
{
   mp_t result = sqrt(x) / 2;
   return boost::math::make_tuple(mp_t(0), x, mp_t(0), result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_x00(mp_t x)
{
   mp_t result = sqrt(x) / 2;
   return boost::math::make_tuple(x, mp_t(0), mp_t(0), result);
}

boost::math::tuple<mp_t, mp_t, mp_t, mp_t> generate_rg_xy0(mp_t x, mp_t y)
{
   mp_t result = rg_imp(x, y, mp_t(0));
   return boost::math::make_tuple(x, y, mp_t(0), result);
}

int cpp_main(int argc, char*argv[])
{
   using namespace boost::math::tools;

   parameter_info<mp_t> arg1, arg2, arg3;
   test_data<mp_t> data;

   bool cont;
   std::string line;

   if(argc < 1)
      return 1;

   do{
#if 0
      int count;
      std::cout << "Number of points: ";
      std::cin >> count;
      
      arg1 = make_periodic_param(mp_t(0), mp_t(1), count);
      arg1.type |= dummy_param;

      //
      // Change this next line to get the R variant you want:
      //
      data.insert(&generate_rd_data, arg1);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
#else
      get_user_parameter_info(arg1, "x");
      get_user_parameter_info(arg2, "y");
      //get_user_parameter_info(arg3, "p");
      arg1.type |= dummy_param;
      arg2.type |= dummy_param;
      //arg3.type |= dummy_param;
      data.insert(generate_rd_data_0xy, arg1, arg2);

      std::cout << "Any more data [y/n]?";
      std::getline(std::cin, line);
      boost::algorithm::trim(line);
      cont = (line == "y");
#endif
   }while(cont);

   std::cout << "Enter name of test data file [default=ellint_rf_data.ipp]";
   std::getline(std::cin, line);
   boost::algorithm::trim(line);
   if(line == "")
      line = "ellint_rf_data.ipp";
   std::ofstream ofs(line.c_str());
   line.erase(line.find('.'));
   ofs << std::scientific << std::setprecision(40);
   write_code(ofs, data, line.c_str());

   return 0;
}


