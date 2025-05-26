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
#include <boost/math/special_functions/hypergeometric_1F1.hpp>
#include <boost/math/special_functions/hypergeometric_pFq.hpp>
#include <boost/math/special_functions/relative_difference.hpp>

#include <boost/random.hpp>
#include <set>
#include <fstream>
#include <boost/iostreams/tee.hpp>
#include <boost/iostreams/stream.hpp>

typedef double test_type;
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

void print_value(double x, std::ostream& os = std::cout)
{
   int e;
   double m = std::frexp(x, &e);
   m = std::ldexp(m, 54);
   e -= 54;
   std::int64_t val = (std::int64_t)m;
   BOOST_MATH_ASSERT(std::ldexp((double)val, e) == x);
   os << "std::ldexp((double)" << val << ", " << e << ")";
}


void print_row(double a, double b, double z, mpfr_float result, std::ostream& os = std::cout)
{
   os << "  {{ ";
   print_value(a, os);
   os << ", ";
   print_value(b, os);
   os << ", ";
   print_value(z, os);
   os << ", SC_(" << std::setprecision(45) << result << ") }}" << std::endl;
}

struct error_data
{
   error_data(double a, double b, double z, std::intmax_t e)
      : a(a), b(b), z(z), error(e) {}
   double a, b, z;
   std::intmax_t error;
   bool operator<(const error_data& other)const
   {
      return error < other.error;
   }
};

int main()
{
   try {
      test_type max_a, max_b, max_z, min_a, min_b, min_z;

      unsigned number_of_samples;

      std::ofstream log_stream, incalculable_stream, unevaluated_stream, bins_stream;
      std::string basename;

      std::cout << "Enter range for a: ";
      std::cin >> min_a >> max_a;
      std::cout << "Enter range for b: ";
      std::cin >> min_b >> max_b;
      std::cout << "Enter range for z: ";
      std::cin >> min_z >> max_z;
      std::cout << "Enter number of samples: ";
      std::cin >> number_of_samples;
      std::cout << "Enter basename for log files: ";
      std::cin >> basename;

      typedef boost::iostreams::tee_device<std::ostream, std::ostream> tee_sink;
      typedef boost::iostreams::stream<tee_sink> tee_stream;

      log_stream.open((basename + ".log").c_str());
      tee_stream tee_log(tee_sink(std::cout, log_stream));
      incalculable_stream.open((basename + "_incalculable.log").c_str());
      unevaluated_stream.open((basename + "_unevaluated.log").c_str());
      bins_stream.open((basename + "_bins.csv").c_str());
      tee_stream tee_bins(tee_sink(std::cout, bins_stream));

      boost::random::mt19937 gen(std::time(0));
      boost::random::uniform_real_distribution<test_type> a_dist(min_a, max_a);
      boost::random::uniform_real_distribution<test_type> b_dist(min_b, max_b);
      boost::random::uniform_real_distribution<test_type> z_dist(min_z, max_z);

      std::multiset<error_data> errors;
      std::map<std::pair<int, int>, int> bins;

      unsigned incalculable = 0;
      unsigned evaluation_errors = 0;
      test_type max_error = 0;

      do
      {
         test_type a = a_dist(gen);
         test_type b = b_dist(gen);
         test_type z = z_dist(gen);
         test_type found, expected;
         mpfr_float mp_expected;

         try {
            mp_expected = boost::math::hypergeometric_pFq_precision({ mpfr_float(a) }, { mpfr_float(b) }, mpfr_float(z), 25, 200.0);
            expected = (test_type)mp_expected;
         }
         catch (const std::exception&)
         {
            // Unable to compute reference value:
            ++incalculable;
            tee_log << "Unable to compute reference value in reasonable time: " << std::endl;
            print_row(a, b, z, mpfr_float(0), tee_log);
            incalculable_stream << std::setprecision(6) << std::scientific << a << "," << b << "," << z << "\n";
            continue;
         }
         try
         {
            found = boost::math::hypergeometric_1F1(a, b, z);
         }
         catch (const std::exception&)
         {
            ++evaluation_errors;
            --number_of_samples;
            log_stream << "Unexpected exception calculating value: " << std::endl;
            print_row(a, b, z, mp_expected, log_stream);
            unevaluated_stream << std::setprecision(6) << std::scientific << a << "," << b << "," << z << "\n";
            continue;
         }
         test_type err = boost::math::epsilon_difference(found, expected);
         if (err > max_error)
         {
            tee_log << "New maximum error is: " << err << std::endl;
            print_row(a, b, z, mp_expected, tee_log);
            max_error = err;
         }
         try {
            errors.insert(error_data(a, b, z, boost::math::lltrunc(err)));
         }
         catch (...)
         {
            errors.insert(error_data(a, b, z, INT_MAX));
         }
         --number_of_samples;
         if (number_of_samples % 500 == 0)
            std::cout << number_of_samples << " samples to go" << std::endl;
      } while (number_of_samples);

      tee_log << "Max error found was: " << max_error << std::endl;

      unsigned current_bin = 0;
      unsigned lim = 1;
      unsigned old_lim = 0;

      while (errors.size())
      {
         old_lim = lim;
         lim *= 2;
         //std::cout << "Enter upper limit for bin " << current_bin << ": ";
         //std::cin >> lim;
         auto p = errors.upper_bound(error_data(0, 0, 0, lim));
         int bin_count = std::distance(errors.begin(), p);
         if (bin_count)
         {
            std::ofstream os((basename + "_errors_" + std::to_string(current_bin + 1) + ".csv").c_str());
            os << "a,b,z,error\n";
            bins[std::make_pair(old_lim, lim)] = bin_count;
            for (auto pos = errors.begin(); pos != p; ++pos)
            {
               os << pos->a << "," << pos->b << "," << pos->z << "," << pos->error << "\n";
            }
            errors.erase(errors.begin(), p);
         }
         ++current_bin;
      }

      tee_bins << "Results:\n\n";
      tee_bins << "#bin,Range,2^N,Count\n";
      int hash = 0;
      for (auto p = bins.begin(); p != bins.end(); ++p, ++hash)
      {
         tee_bins << hash << "," << p->first.first << "-" << p->first.second << "," << hash+1 << "," << p->second << std::endl;
      }
      if (evaluation_errors)
      {
         tee_bins << ",Failed,," << evaluation_errors << std::endl;
      }
      if (incalculable)
      {
         tee_bins << ",Incalculable,," << incalculable << std::endl;
      }
   }
   catch (const std::exception& e)
   {
      std::cout << "Terminating with unhandled exception: " << e.what() << std::endl;
   }

   return 0;
}

