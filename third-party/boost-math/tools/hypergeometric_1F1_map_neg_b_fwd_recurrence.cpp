//  Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_ENABLE_ASSERT_HANDLER
#define BOOST_MATH_MAX_SERIES_ITERATION_POLICY INT_MAX
// for consistent behaviour across compilers/platforms:
#define BOOST_MATH_PROMOTE_DOUBLE_POLICY false
// overflow to infinity is OK, we treat these as zero error as long as the sign is correct!
#define BOOST_MATH_OVERFLOW_ERROR_POLICY ignore_error

#include <iostream>
#include <ctime>
#include <boost/multiprecision/mpfr.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/special_functions/relative_difference.hpp>

#include <boost/random.hpp>
#include <set>
#include <fstream>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

using boost::multiprecision::mpfr_float;

namespace boost {
   //
   // We convert assertions into exceptions, so we can log them and continue:
   //
   void assertion_failed(char const * expr, char const *, char const * file, long line)
   {
      std::ostringstream oss;
      oss << file << ":" << line << " Assertion failed: " << expr;
      throw std::runtime_error(oss.str());
   }

}

typedef boost::multiprecision::cpp_bin_float_quad test_type;

int main()
{
   using std::floor;
   using std::ceil;
   try {
      test_type a_start, a_end;
      test_type b_start, b_end;
      test_type a_mult, b_mult;

      std::cout << "Enter range for parameter a: ";
      std::cin >> a_start >> a_end;
      std::cout << "Enter range for parameter b: ";
      std::cin >> b_start >> b_end;
      std::cout << "Enter multiplier for a parameter: ";
      std::cin >> a_mult;
      std::cout << "Enter multiplier for b parameter: ";
      std::cin >> b_mult;

      double error_limit = 200;
      double time_limit = 10.0;

      for (test_type a = a_start; a < a_end; a_start < 0 ? a /= a_mult : a *= a_mult)
      {
         for (test_type b = b_start; b < b_end; b_start < 0 ? b /= b_mult : b *= b_mult)
         {
            test_type z_mult = 2;
            test_type last_good = 0;
            test_type bad = 0;
            try {
               for (test_type z = 1; z < 1e10; z *= z_mult, z_mult *= 2)
               {
                  // std::cout << "z = " << z << std::endl;
                  std::uintmax_t max_iter = 1000;
                  test_type calc = boost::math::tools::function_ratio_from_forwards_recurrence(boost::math::detail::hypergeometric_1F1_recurrence_a_and_b_coefficients<test_type>(a, b, z), std::numeric_limits<test_type>::epsilon() * 2, max_iter);
                  test_type reference = (test_type)(boost::math::hypergeometric_pFq_precision({ mpfr_float(a) }, { mpfr_float(b) }, mpfr_float(z), 50, time_limit) / boost::math::hypergeometric_pFq_precision({ mpfr_float(a + 1) }, { mpfr_float(b + 1) }, mpfr_float(z), std::numeric_limits<test_type>::digits10 * 2, time_limit));
                  double err = (double)boost::math::epsilon_difference(reference, calc);

                  if (err < error_limit)
                  {
                     last_good = z;
                     break;
                  }
                  else
                  {
                     bad = z;
                  }
               }
            }
            catch (const std::exception& e)
            {
               std::cout << "Unexpected exception: " << e.what() << std::endl;
               std::cout << "For a = " << a << " b = " << b << " z = " << bad * z_mult / 2 << std::endl;
            }
            test_type z_limit;
            if (0 == bad)
               z_limit = 1;  // Any z is large enough
            else if (0 == last_good)
               z_limit = std::numeric_limits<test_type > ::infinity();
            else
            {
               //
               // At this stage last_good and bad should bracket the edge of the domain, bisect to narrow things down:
               //
               z_limit = last_good == 0 ? 0 : boost::math::tools::bisect([&a, b, error_limit, time_limit](test_type z)
               {
                  std::uintmax_t max_iter = 1000;
                  test_type calc = boost::math::tools::function_ratio_from_forwards_recurrence(boost::math::detail::hypergeometric_1F1_recurrence_a_and_b_coefficients<test_type>(a, b, z), std::numeric_limits<test_type>::epsilon() * 2, max_iter);
                  test_type reference = (test_type)(boost::math::hypergeometric_pFq_precision({ mpfr_float(a) }, { mpfr_float(b) }, mpfr_float(z), 50, time_limit + 20) / boost::math::hypergeometric_pFq_precision({ mpfr_float(a + 1) }, { mpfr_float(b + 1) }, mpfr_float(z), std::numeric_limits<test_type>::digits10 * 2, time_limit + 20));
                  test_type err = boost::math::epsilon_difference(reference, calc);
                  return err < error_limit ? 1 : -1;
               }, bad, last_good, boost::math::tools::equal_floor()).first;
               z_limit = floor(z_limit + 2);  // Give ourselves some headroom!
            }
            // std::cout << "z_limit = " << z_limit << std::endl;
            //
            // Now over again for backwards recurrence domain at the same points:
            //
            bad = z_limit > 1e10 ? 1e10 : z_limit;
            last_good = 0;
            z_mult = 1.1;
            for (test_type z = bad; z > 1; z /= z_mult, z_mult *= 2)
            {
               // std::cout << "z = " << z << std::endl;
               try {
                  std::uintmax_t max_iter = 1000;
                  test_type calc = boost::math::tools::function_ratio_from_backwards_recurrence(boost::math::detail::hypergeometric_1F1_recurrence_a_and_b_coefficients<test_type>(a, b, z), std::numeric_limits<test_type>::epsilon() * 2, max_iter);
                  test_type reference = (test_type)(boost::math::hypergeometric_pFq_precision({ mpfr_float(a) }, { mpfr_float(b) }, mpfr_float(z), 50, time_limit) / boost::math::hypergeometric_pFq_precision({ mpfr_float(a - 1) }, { mpfr_float(b - 1) }, mpfr_float(z), std::numeric_limits<test_type>::digits10 * 2, time_limit));
                  test_type err = boost::math::epsilon_difference(reference, calc);

                  if (err < error_limit)
                  {
                     last_good = z;
                     break;
                  }
                  else
                  {
                     bad = z;
                  }
               }
               catch (const std::exception& e)
               {
                  bad = z;
                  std::cout << "Unexpected exception: " << e.what() << std::endl;
                  std::cout << "For a = " << a << " b = " << b << " z = " << z << std::endl;
               }
            }
            test_type lower_z_limit;
            if (last_good < 1)
               lower_z_limit = 0;
            else if (last_good >= bad)
            {
               std::uintmax_t max_iter = 1000;
               test_type z = bad;
               test_type calc = boost::math::tools::function_ratio_from_forwards_recurrence(boost::math::detail::hypergeometric_1F1_recurrence_a_and_b_coefficients<test_type>(a, b, z), std::numeric_limits<test_type>::epsilon() * 2, max_iter);
               test_type reference = (test_type)(boost::math::hypergeometric_pFq_precision({ mpfr_float(a) }, { mpfr_float(b) }, mpfr_float(z), 50, time_limit) / boost::math::hypergeometric_pFq_precision({ mpfr_float(a + 1) }, { mpfr_float(b + 1) }, mpfr_float(z), std::numeric_limits<test_type>::digits10 * 2, time_limit));
               test_type err = boost::math::epsilon_difference(reference, calc);
               if (err < error_limit)
               {
                  lower_z_limit = bad;   //  Both forwards and backwards iteration work!!!
               }
               else
                  throw std::runtime_error("Internal logic failed!");
            }
            else
            {
               //
               // At this stage last_good and bad should bracket the edge of the domain, bisect to narrow things down:
               //
               lower_z_limit = last_good == 0 ? 0 : boost::math::tools::bisect([&a, b, error_limit, time_limit](test_type z)
               {
                  std::uintmax_t max_iter = 1000;
                  test_type calc = boost::math::tools::function_ratio_from_backwards_recurrence(boost::math::detail::hypergeometric_1F1_recurrence_a_and_b_coefficients<test_type>(a, b, z), std::numeric_limits<test_type>::epsilon() * 2, max_iter);
                  test_type reference = (test_type)(boost::math::hypergeometric_pFq_precision({ mpfr_float(a) }, { mpfr_float(b) }, mpfr_float(z), 50, time_limit + 20) / boost::math::hypergeometric_pFq_precision({ mpfr_float(a - 1) }, { mpfr_float(b - 1) }, mpfr_float(z), std::numeric_limits<test_type>::digits10 * 2, time_limit + 20));
                  test_type err = boost::math::epsilon_difference(reference, calc);
                  return err < error_limit ? 1 : -1;
               }, last_good, bad, boost::math::tools::equal_floor()).first;
               z_limit = ceil(z_limit - 2);  // Give ourselves some headroom!
            }

            std::cout << std::setprecision(std::numeric_limits<test_type>::max_digits10) << "{ " << a << ", " << b << ", " << lower_z_limit << ", " << z_limit << "}," << std::endl;
         }
      }
   }
   catch (const std::exception& e)
   {
      std::cout << "Unexpected exception: " << e.what() << std::endl;
   }
   return 0;
}

