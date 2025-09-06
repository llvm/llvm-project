// Copyright John Maddock 2023.

// Use, modification and distribution are subject to the
// Boost Software License, Version 1.0.
// (See accompanying file LICENSE_1_0.txt
// or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifdef _MSC_VER
#define _SCL_SECURE_NO_WARNINGS
#endif

#include <boost/cstdfloat.hpp>
#include <boost/math/special_functions/next.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <boost/detail/lightweight_test.hpp>
#include <array>
#include <iostream>
#include <sstream>
#include <iomanip>

#ifdef BOOST_FLOAT128_C
#if defined(__INTEL_COMPILER) || defined(BOOST_MATH_TEST_IO_AS_INTEL_QUAD)
bool has_bad_bankers_rounding(const boost::float128_t&)
{
   return true;
}
#endif
#endif

template <class T>
bool has_bad_bankers_rounding(const T&)
{
   return false;
}

enum
{
   warn_on_fail,
   error_on_fail,
   abort_on_fail
};

inline std::ostream& report_where(const char* file, int line, const char* function)
{
   if (function)
      BOOST_LIGHTWEIGHT_TEST_OSTREAM << "In function: " << function << std::endl;
   BOOST_LIGHTWEIGHT_TEST_OSTREAM << file << ":" << line;
   return BOOST_LIGHTWEIGHT_TEST_OSTREAM;
}

#define BOOST_MP_REPORT_WHERE report_where(__FILE__, __LINE__, BOOST_CURRENT_FUNCTION)

inline void report_severity(int severity)
{
   if (severity == error_on_fail)
      ++boost::detail::test_errors();
   else if (severity == abort_on_fail)
   {
      ++boost::detail::test_errors();
      abort();
   }
}

#define BOOST_MP_REPORT_SEVERITY(severity) report_severity(severity)

template <class E>
void report_unexpected_exception(const E& e, int severity, const char* file, int line, const char* function)
{
   report_where(file, line, function) << " Unexpected exception of type " << typeid(e).name() << std::endl;
   BOOST_LIGHTWEIGHT_TEST_OSTREAM << "Errot message was: " << e.what() << std::endl;
   BOOST_MP_REPORT_SEVERITY(severity);
}

#ifdef BOOST_HAS_INT128

std::ostream& operator<<(std::ostream& os, boost::int128_type val)
{
   std::stringstream ss;
   ss << std::hex << "0x" << static_cast<std::uint64_t>(static_cast<boost::uint128_type>(val) >> 64) << static_cast<std::uint64_t>(val);
   return os << ss.str();
}

std::ostream& operator<<(std::ostream& os, boost::uint128_type val)
{
   std::stringstream ss;
   ss << std::hex << "0x" << static_cast<std::uint64_t>(val >> 64) << static_cast<std::uint64_t>(val);
   return os << ss.str();
}

#endif

#ifndef BOOST_NO_EXCEPTIONS
#define BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)                                       \
   catch (const std::exception& e)                                                          \
   {                                                                                        \
      report_unexpected_exception(e, severity, __FILE__, __LINE__, BOOST_CURRENT_FUNCTION); \
   }                                                                                        \
   catch (...)                                                                              \
   {                                                                                        \
      std::cout << "Exception of unknown type was thrown" << std::endl;                     \
      report_severity(severity);                                                            \
   }
#define BOOST_MP_TEST_TRY try
#else
#define BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)
#define BOOST_MP_TEST_TRY
#endif

#define BOOST_CHECK_IMP(x, severity)                                                        \
   BOOST_MP_TEST_TRY                                                                             \
   {                                                                                        \
      if (x)                                                                                \
      {                                                                                     \
      }                                                                                     \
      else                                                                                  \
      {                                                                                     \
         BOOST_MP_REPORT_WHERE << " Failed predicate: " << BOOST_STRINGIZE(x) << std::endl; \
         BOOST_MP_REPORT_SEVERITY(severity);                                                \
      }                                                                                     \
   }                                                                                        \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#define BOOST_CHECK(x) BOOST_CHECK_IMP(x, error_on_fail)
#define BOOST_WARN(x) BOOST_CHECK_IMP(x, warn_on_fail)
#define BOOST_REQUIRE(x) BOOST_CHECK_IMP(x, abort_on_fail)

#define BOOST_CLOSE_IMP(x, y, tol, severity)                                                \
   BOOST_MP_TEST_TRY                                                                             \
   {                                                                                        \
      if (relative_error(x, y) > tol)                                                       \
      {                                                                                     \
         BOOST_MP_REPORT_WHERE << " Failed check for closeness: \n"                         \
                               << std::setprecision(std::numeric_limits<decltype(x)>::max_digits10) << std::scientific        \
                               << "Value of LHS was: " << x << "\n"                         \
                               << "Value of RHS was: " << y << "\n"                         \
                               << std::setprecision(5) << std::fixed                        \
                               << "Relative error was: " << relative_error(x, y) << "eps\n" \
                               << "Tolerance was: " << tol << "eps" << std::endl;           \
         BOOST_MP_REPORT_SEVERITY(severity);                                                \
      }                                                                                     \
   }                                                                                        \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#define BOOST_EQUAL_IMP(x, y, severity)                                              \
   BOOST_MP_TEST_TRY                                                                      \
   {                                                                                 \
      if (!((x) == (y)))                                                             \
      {                                                                              \
         BOOST_MP_REPORT_WHERE << " Failed check for equality: \n"                   \
                               << std::setprecision(std::numeric_limits<decltype(x)>::max_digits10) << std::scientific \
                               << "Value of LHS was: " << (x) << "\n"                \
                               << "Value of RHS was: " << (y) << "\n"                \
                               << std::setprecision(3) << std::endl;                 \
         BOOST_MP_REPORT_SEVERITY(severity);                                         \
      }                                                                              \
   }                                                                                 \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#define BOOST_NE_IMP(x, y, severity)                                                 \
   BOOST_MP_TEST_TRY                                                                      \
   {                                                                                 \
      if (!(x != y))                                                                 \
      {                                                                              \
         BOOST_MP_REPORT_WHERE << " Failed check for non-equality: \n"               \
                               << std::setprecision(std::numeric_limits<decltype(x)>::max_digits10) << std::scientific \
                               << "Value of LHS was: " << x << "\n"                  \
                               << "Value of RHS was: " << y << "\n"                  \
                               << std::setprecision(3) << std::endl;                 \
         BOOST_MP_REPORT_SEVERITY(severity);                                         \
      }                                                                              \
   }                                                                                 \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#define BOOST_LT_IMP(x, y, severity)                                                 \
   BOOST_MP_TEST_TRY                                                                      \
   {                                                                                 \
      if (!(x < y))                                                                  \
      {                                                                              \
         BOOST_MP_REPORT_WHERE << " Failed check for less than: \n"                  \
                               << std::setprecision(std::numeric_limits<decltype(x)>::max_digits10) << std::scientific \
                               << "Value of LHS was: " << x << "\n"                  \
                               << "Value of RHS was: " << y << "\n"                  \
                               << std::setprecision(3) << std::endl;                 \
         BOOST_MP_REPORT_SEVERITY(severity);                                         \
      }                                                                              \
   }                                                                                 \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#define BOOST_GT_IMP(x, y, severity)                                                 \
   BOOST_MP_TEST_TRY                                                                      \
   {                                                                                 \
      if (!(x > y))                                                                  \
      {                                                                              \
         BOOST_MP_REPORT_WHERE << " Failed check for greater than: \n"               \
                               << std::setprecision(std::numeric_limits<decltype(x)>::max_digits10) << std::scientific \
                               << "Value of LHS was: " << x << "\n"                  \
                               << "Value of RHS was: " << y << "\n"                  \
                               << std::setprecision(3) << std::endl;                 \
         BOOST_MP_REPORT_SEVERITY(severity);                                         \
      }                                                                              \
   }                                                                                 \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#define BOOST_LE_IMP(x, y, severity)                                                 \
   BOOST_MP_TEST_TRY                                                                      \
   {                                                                                 \
      if (!(x <= y))                                                                 \
      {                                                                              \
         BOOST_MP_REPORT_WHERE << " Failed check for less-than-equal-to: \n"         \
                               << std::setprecision(std::numeric_limits<decltype(x)>::max_digits10) << std::scientific \
                               << "Value of LHS was: " << x << "\n"                  \
                               << "Value of RHS was: " << y << "\n"                  \
                               << std::setprecision(3) << std::endl;                 \
         BOOST_MP_REPORT_SEVERITY(severity);                                         \
      }                                                                              \
   }                                                                                 \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#define BOOST_GE_IMP(x, y, severity)                                                 \
   BOOST_MP_TEST_TRY                                                                      \
   {                                                                                 \
      if (!(x >= y))                                                                 \
      {                                                                              \
         BOOST_MP_REPORT_WHERE << " Failed check for greater-than-equal-to \n"       \
                               << std::setprecision(std::numeric_limits<decltype(x)>::max_digits10) << std::scientific \
                               << "Value of LHS was: " << x << "\n"                  \
                               << "Value of RHS was: " << y << "\n"                  \
                               << std::setprecision(3) << std::endl;                 \
         BOOST_MP_REPORT_SEVERITY(severity);                                         \
      }                                                                              \
   }                                                                                 \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)

#ifndef BOOST_NO_EXCEPTIONS
#define BOOST_MT_CHECK_THROW_IMP(x, E, severity)                                                                   \
   BOOST_MP_TEST_TRY                                                                                                    \
   {                                                                                                               \
      x;                                                                                                           \
      BOOST_MP_REPORT_WHERE << " Expected exception not thrown in expression " << BOOST_STRINGIZE(x) << std::endl; \
      BOOST_MP_REPORT_SEVERITY(severity);                                                                          \
   }                                                                                                               \
   catch (const E&) {}                                                                                             \
   BOOST_MP_UNEXPECTED_EXCEPTION_CHECK(severity)
#else
#define BOOST_MT_CHECK_THROW_IMP(x, E, severity)
#endif

#define BOOST_CHECK_EQUAL(x, y) BOOST_EQUAL_IMP(x, y, error_on_fail)
#define BOOST_WARN_EQUAL(x, y) BOOST_EQUAL_IMP(x, y, warn_on_fail)
#define BOOST_REQUIRE_EQUAL(x, y) BOOST_EQUAL_IMP(x, y, abort_on_fail)

void print_flags(std::ios_base::fmtflags f)
{
   std::cout << "Formatting flags were: ";
   if (f & std::ios_base::scientific)
      std::cout << "scientific ";
   if (f & std::ios_base::fixed)
      std::cout << "fixed ";
   if (f & std::ios_base::showpoint)
      std::cout << "showpoint ";
   if (f & std::ios_base::showpos)
      std::cout << "showpos ";
   std::cout << std::endl;
}

bool is_bankers_rounding_error(const std::string& s, const char* expect)
{
   // This check isn't foolproof: that would require *much* more sophisticated code!!!
   std::string::size_type l = std::strlen(expect);
   if (l != s.size())
      return false;
   std::string::size_type len = s.find('e');
   if (len == std::string::npos)
      len = l - 1;
   else
      --len;
   if (s.compare(0, len, expect, len))
      return false;
   if (s[len] != expect[len] + 1)
      return false;
   return true;
}

template <class T>
bool is_bankers_rounding_error(T new_val, T val)
{
   // This check isn't foolproof: that would require *much* more sophisticated code!!!
   auto n = boost::math::float_distance(new_val, val) == 1;
   std::cout << "Distance was: " << n << std::endl;
   return std::abs(n) <= 1;
}

template <class T>
void test()
{
   typedef T                                mp_t;
   std::array<std::ios_base::fmtflags, 9> f =
       {{std::ios_base::fmtflags(0), std::ios_base::showpoint, std::ios_base::showpos, std::ios_base::scientific, std::ios_base::scientific | std::ios_base::showpos,
         std::ios_base::scientific | std::ios_base::showpoint, std::ios_base::fixed, std::ios_base::fixed | std::ios_base::showpoint,
         std::ios_base::fixed | std::ios_base::showpos}};

   std::array<std::array<const char*, 13 * 9>, 40> string_data = {{
#include "string_data.ipp"
   }};

   double num   = 123456789.0;
   double denom = 1;
   double val   = num;
   for (unsigned j = 0; j < 40; ++j)
   {
      unsigned col = 0;
      for (unsigned prec = 1; prec < 14; ++prec)
      {
         for (unsigned i = 0; i < f.size(); ++i, ++col)
         {
            std::stringstream ss;
            ss.precision(prec);
            ss.flags(f[i]);
            ss << mp_t(val);
            const char* expect = string_data[j][col];
            if (ss.str() != expect)
            {
               if (has_bad_bankers_rounding(mp_t()) && is_bankers_rounding_error(ss.str(), expect))
               {
                  std::cout << "Ignoring bankers-rounding error with Intel _Quad.\n";
               }
               else
               {
                  std::cout << std::setprecision(20) << "Testing value " << val << std::endl;
                  print_flags(f[i]);
                  std::cout << "Precision is: " << prec << std::endl;
                  std::cout << "Got:      " << ss.str() << std::endl;
                  std::cout << "Expected: " << expect << std::endl;
                  ++boost::detail::test_errors();
               }
            }
         }
      }
      num = -num;
      if (j & 1)
         denom *= 8;
      val = num / denom;
   }

   std::array<const char*, 13 * 9> zeros =
       {{"0", "0.", "+0", "0.0e+00", "+0.0e+00", "0.0e+00", "0.0", "0.0", "+0.0", "0", "0.0", "+0", "0.00e+00", "+0.00e+00", "0.00e+00", "0.00", "0.00", "+0.00", "0", "0.00", "+0", "0.000e+00", "+0.000e+00", "0.000e+00", "0.000", "0.000", "+0.000", "0", "0.000", "+0", "0.0000e+00", "+0.0000e+00", "0.0000e+00", "0.0000", "0.0000", "+0.0000", "0", "0.0000", "+0", "0.00000e+00", "+0.00000e+00", "0.00000e+00", "0.00000", "0.00000", "+0.00000", "0", "0.00000", "+0", "0.000000e+00", "+0.000000e+00", "0.000000e+00", "0.000000", "0.000000", "+0.000000", "0", "0.000000", "+0", "0.0000000e+00", "+0.0000000e+00", "0.0000000e+00", "0.0000000", "0.0000000", "+0.0000000", "0", "0.0000000", "+0", "0.00000000e+00", "+0.00000000e+00", "0.00000000e+00", "0.00000000", "0.00000000", "+0.00000000", "0", "0.00000000", "+0", "0.000000000e+00", "+0.000000000e+00", "0.000000000e+00", "0.000000000", "0.000000000", "+0.000000000", "0", "0.000000000", "+0", "0.0000000000e+00", "+0.0000000000e+00", "0.0000000000e+00", "0.0000000000", "0.0000000000", "+0.0000000000", "0", "0.0000000000", "+0", "0.00000000000e+00", "+0.00000000000e+00", "0.00000000000e+00", "0.00000000000", "0.00000000000", "+0.00000000000", "0", "0.00000000000", "+0", "0.000000000000e+00", "+0.000000000000e+00", "0.000000000000e+00", "0.000000000000", "0.000000000000", "+0.000000000000", "0", "0.000000000000", "+0", "0.0000000000000e+00", "+0.0000000000000e+00", "0.0000000000000e+00", "0.0000000000000", "0.0000000000000", "+0.0000000000000"}};

   unsigned col = 0;
   val          = 0;
   for (unsigned prec = 1; prec < 14; ++prec)
   {
      for (unsigned i = 0; i < f.size(); ++i, ++col)
      {
         std::stringstream ss;
         ss.precision(prec);
         ss.flags(f[i]);
         ss << mp_t(val);
         const char* expect = zeros[col];
         if (ss.str() != expect)
         {
            std::cout << std::setprecision(20) << "Testing value " << val << std::endl;
            print_flags(f[i]);
            std::cout << "Precision is: " << prec << std::endl;
            std::cout << "Got:      " << ss.str() << std::endl;
            std::cout << "Expected: " << expect << std::endl;
            ++boost::detail::test_errors();
         }
      }
   }
}

template <class T>
T generate_random()
{
   typedef int                                     e_type;
   static boost::random::mt19937                   gen;
   T                                               val      = gen();
   T                                               prev_val = -1;
   while (val != prev_val)
   {
      val *= (gen.max)();
      prev_val = val;
      val += gen();
   }
   e_type e;
   val = std::frexp(val, &e);

   static boost::random::uniform_int_distribution<e_type> ui(0, std::numeric_limits<T>::max_exponent - 10);
   return std::ldexp(val, ui(gen));
}

template <class T>
void do_round_trip(const T& val, std::ios_base::fmtflags f)
{
   std::stringstream ss;
   ss << std::setprecision(std::numeric_limits<T>::max_digits10);
   ss.flags(f);
   ss << val;
   T new_val;
   ss >> new_val;
   if (new_val != val)
   {
      if (has_bad_bankers_rounding(T()) && is_bankers_rounding_error(new_val, val))
      {
         std::cout << "Ignoring bankers-rounding error with Intel _Quad mp_f.\n";
      }
      else
      {
         BOOST_CHECK_EQUAL(new_val, val);
      }
   }
}

template <class T>
void do_round_trip(const T& val)
{
   do_round_trip(val, std::ios_base::fmtflags(0));
   do_round_trip(val, std::ios_base::fmtflags(std::ios_base::scientific));
   if ((fabs(val) > 1) && (fabs(val) < 1e100))
      do_round_trip(val, std::ios_base::fmtflags(std::ios_base::fixed));
}

template <class T>
void test_round_trip()
{
   for (unsigned i = 0; i < 1000; ++i)
   {
      T val = generate_random<T>();
      do_round_trip(val);
      do_round_trip(T(-val));
      do_round_trip(T(1 / val));
      do_round_trip(T(-1 / val));
   }
}

int main()
{
   test<double>();
   test<boost::float64_t>();

   test_round_trip<double>();
   test_round_trip<boost::float64_t>();

#ifdef BOOST_FLOAT128_C
   test<boost::float128_t>();
   test_round_trip<boost::float128_t>();
#endif
   return boost::report_errors();
}
