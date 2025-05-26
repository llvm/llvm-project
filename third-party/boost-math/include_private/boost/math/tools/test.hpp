//  (C) Copyright John Maddock 2006.
//  (C) Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TOOLS_TEST_HPP
#define BOOST_MATH_TOOLS_TEST_HPP

#ifdef _MSC_VER
#pragma once
#endif

#include <boost/math/tools/config.hpp>
#include <boost/math/tools/stats.hpp>
#include <boost/math/special_functions/fpclassify.hpp>
#include <boost/math/special_functions/relative_difference.hpp>
#include <boost/math/policies/error_handling.hpp>
#include <boost/test/test_tools.hpp>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace boost{ namespace math{ namespace tools{

template <class T>
struct test_result
{
private:
   boost::math::tools::stats<T> stat;   // Statistics for the test.
   unsigned worst_case;                 // Index of the worst case test.
public:
   test_result() { worst_case = 0; }
   void set_worst(int i){ worst_case = i; }
   void add(const T& point){ stat.add(point); }
   // accessors:
   unsigned worst()const{ return worst_case; }
   T min BOOST_MATH_PREVENT_MACRO_SUBSTITUTION()const{ return (stat.min)(); }
   T max BOOST_MATH_PREVENT_MACRO_SUBSTITUTION()const{ return (stat.max)(); }
   T total()const{ return stat.total(); }
   T mean()const{ return stat.mean(); }
   std::uintmax_t count()const{ return stat.count(); }
   T variance()const{ return stat.variance(); }
   T variance1()const{ return stat.variance1(); }
   T rms()const{ return stat.rms(); }

   test_result& operator+=(const test_result& t)
   {
      if((t.stat.max)() > (stat.max)())
         worst_case = t.worst_case;
      stat += t.stat;
      return *this;
   }
};

template <class T>
struct calculate_result_type
{
   typedef typename T::value_type row_type;
   typedef typename row_type::value_type value_type;
};

template <class T>
T relative_error(T a, T b)
{
   return boost::math::relative_difference(a, b);
}


template <class T>
void set_output_precision(T, std::ostream& os)
{
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable:4127)
#endif
   if(std::numeric_limits<T>::digits10)
   {
      os << std::setprecision(std::numeric_limits<T>::digits10 + 2);
   }
   else
      os << std::setprecision(22); // and hope for the best!

#ifdef _MSC_VER
#pragma warning(pop)
#endif
}

template <class Seq>
void print_row(const Seq& row, std::ostream& os = std::cout)
{
   try {
      set_output_precision(row[0], os);
      for (unsigned i = 0; i < row.size(); ++i)
      {
         if (i)
            os << ", ";
         os << row[i];
      }
      os << std::endl;
   }
   catch (const std::exception&) {}
}

//
// Function test accepts an matrix of input values (probably a 2D std::array)
// and calls two functors for each row in the array - one calculates a value
// to test, and one extracts the expected value from the array (or possibly
// calculates it at high precision).  The two functors are usually simple lambda
// expressions.
//
template <class A, class F1, class F2>
test_result<typename calculate_result_type<A>::value_type> test(const A& a, F1 test_func, F2 expect_func)
{
   typedef typename A::value_type         row_type;
   typedef typename row_type::value_type  value_type;

   test_result<value_type> result;

   for(unsigned i = 0; i < a.size(); ++i)
   {
      const row_type& row = a[i];
      value_type point;
#ifndef BOOST_NO_EXCEPTIONS
      try
      {
#endif
         point = test_func(row);
#ifndef BOOST_NO_EXCEPTIONS
      }
      catch(const std::underflow_error&)
      {
         point = 0;
      }
      catch(const std::overflow_error&)
      {
         point = std::numeric_limits<value_type>::has_infinity ?
            std::numeric_limits<value_type>::infinity()
            : tools::max_value<value_type>();
      }
      catch(const std::exception& e)
      {
         std::cerr << e.what() << std::endl;
         print_row(row, std::cerr);
         BOOST_ERROR("Unexpected exception.");
         // so we don't get further errors:
         point = expect_func(row);
      }
#endif
      value_type expected = expect_func(row);
      value_type err = relative_error(point, expected);
#ifdef BOOST_INSTRUMENT
      if(err != 0)
      {
         std::cout << row[0] << " " << err;
         if(std::numeric_limits<value_type>::is_specialized)
         {
            std::cout << " (" << err / std::numeric_limits<value_type>::epsilon() << "eps)";
         }
         std::cout << std::endl;
      }
#endif
      if(!(boost::math::isfinite)(point) && (boost::math::isfinite)(expected))
      {
         std::cerr << "CAUTION: Found non-finite result, when a finite value was expected at entry " << i << "\n";
         std::cerr << "Found: " << point << " Expected " << expected << " Error: " << err << std::endl;
         print_row(row, std::cerr);
         BOOST_ERROR("Unexpected non-finite result");
      }
      if(err > 0.5)
      {
         std::cerr << "CAUTION: Gross error found at entry " << i << ".\n";
         std::cerr << "Found: " << point << " Expected " << expected << " Error: " << err << std::endl;
         print_row(row, std::cerr);
         BOOST_ERROR("Gross error");
      }
      result.add(err);
      if((result.max)() == err)
         result.set_worst(i);
   }
   return result;
}

template <class Real, class A, class F1, class F2>
test_result<Real> test_hetero(const A& a, F1 test_func, F2 expect_func)
{
   typedef typename A::value_type         row_type;
   typedef Real                          value_type;

   test_result<value_type> result;

   for(unsigned i = 0; i < a.size(); ++i)
   {
      const row_type& row = a[i];
      value_type point;
#ifndef BOOST_NO_EXCEPTIONS
      try
      {
#endif
         point = test_func(row);
#ifndef BOOST_NO_EXCEPTIONS
      }
      catch(const std::underflow_error&)
      {
         point = 0;
      }
      catch(const std::overflow_error&)
      {
         point = std::numeric_limits<value_type>::has_infinity ?
            std::numeric_limits<value_type>::infinity()
            : tools::max_value<value_type>();
      }
      catch(const std::exception& e)
      {
         std::cerr << "Unexpected exception at entry: " << i << "\n";
         std::cerr << e.what() << std::endl;
         print_row(row, std::cerr);
         BOOST_ERROR("Unexpected exception.");
         // so we don't get further errors:
         point = expect_func(row);
      }
#endif
      value_type expected = expect_func(row);
      value_type err = relative_error(point, expected);
#ifdef BOOST_INSTRUMENT
      if(err != 0)
      {
         std::cout << row[0] << " " << err;
         if(std::numeric_limits<value_type>::is_specialized)
         {
            std::cout << " (" << err / std::numeric_limits<value_type>::epsilon() << "eps)";
         }
         std::cout << std::endl;
      }
#endif
      if(!(boost::math::isfinite)(point) && (boost::math::isfinite)(expected))
      {
         std::cerr << "CAUTION: Found non-finite result, when a finite value was expected at entry " << i << "\n";
         std::cerr << "Found: " << point << " Expected " << expected << " Error: " << err << std::endl;
         print_row(row, std::cerr);
         BOOST_ERROR("Unexpected non-finite result");
      }
      if(err > 0.5)
      {
         std::cerr << "CAUTION: Gross error found at entry " << i << ".\n";
         std::cerr << "Found: " << point << " Expected " << expected << " Error: " << err << std::endl;
         print_row(row, std::cerr);
         BOOST_ERROR("Gross error");
      }
      result.add(err);
      if((result.max)() == err)
         result.set_worst(i);
   }
   return result;
}

#ifndef BOOST_MATH_NO_EXCEPTIONS
template <class Val, class Exception>
void test_check_throw(Val, Exception)
{
   BOOST_CHECK(errno);
   errno = 0;
}

template <class Val>
void test_check_throw(Val val, std::domain_error const*)
{
   BOOST_CHECK(errno == EDOM);
   errno = 0;
   if(std::numeric_limits<Val>::has_quiet_NaN)
   {
      BOOST_CHECK((boost::math::isnan)(val));
   }
}

template <class Val>
void test_check_throw(Val v, std::overflow_error const*)
{
   BOOST_CHECK(errno == ERANGE);
   errno = 0;
   BOOST_CHECK((v >= boost::math::tools::max_value<Val>()) || (v <= -boost::math::tools::max_value<Val>()));
}

template <class Val>
void test_check_throw(Val v, boost::math::rounding_error const*)
{
   BOOST_CHECK(errno == ERANGE);
   errno = 0;
   if(std::numeric_limits<Val>::is_specialized && std::numeric_limits<Val>::is_integer)
   {
      BOOST_CHECK((v == (std::numeric_limits<Val>::max)()) || (v == (std::numeric_limits<Val>::min)()));
   }
   else
   {
      BOOST_CHECK((v == boost::math::tools::max_value<Val>()) || (v == -boost::math::tools::max_value<Val>()));
   }
}
#endif

} // namespace tools
} // namespace math
} // namespace boost


  //
  // exception-free testing support, ideally we'd only define this in our tests,
  // but to keep things simple we really need it somewhere that's always included:
  //
#if defined(BOOST_MATH_NO_EXCEPTIONS) && defined(BOOST_MATH_HAS_GPU_SUPPORT)
#  define BOOST_MATH_CHECK_THROW(x, y)
#elif defined(BOOST_MATH_NO_EXCEPTIONS) 
#  define BOOST_MATH_CHECK_THROW(x, ExceptionType) boost::math::tools::test_check_throw(x, static_cast<ExceptionType const*>(nullptr));
#else
#  define BOOST_MATH_CHECK_THROW(x, y) BOOST_CHECK_THROW(x, y)
#endif

#endif


