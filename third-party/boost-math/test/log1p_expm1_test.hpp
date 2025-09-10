//  Copyright John Maddock 2005.
//  Copyright Paul A. Bristow 2010
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <boost/array.hpp>
#include "functor.hpp"

#include "handle_test_result.hpp"
#include "table_type.hpp"


template <class Real, class T>
void do_test(const T& data, const char* type_name, const char* test_name)
{
   typedef Real value_type;

   typedef value_type (*pg)(value_type);
#ifdef LOG1P_FUNCTION_TO_TEST
   pg funcp = LOG1P_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   pg funcp = &boost::math::log1p<value_type>;
#else
   pg funcp = &boost::math::log1p;
#endif

   boost::math::tools::test_result<value_type> result;
   std::cout << "Testing " << test_name << " with type " << type_name
      << "\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n";
   //
   // test log1p against data:
   //
#if !(defined(ERROR_REPORTING_MODE) && !defined(LOG1P_FUNCTION_TO_TEST))
   result = boost::math::tools::test_hetero<Real>(
      data, 
         bind_func<Real>(funcp, 0), 
         extract_result<Real>(1));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "log1p", "Random test data");
   std::cout << std::endl;
#endif
#if !(defined(ERROR_REPORTING_MODE) && !defined(EXPM1_FUNCTION_TO_TEST))
   //
   // test expm1 against data:
   //
#ifdef EXPM1_FUNCTION_TO_TEST
   funcp = EXPM1_FUNCTION_TO_TEST;
#elif defined(BOOST_MATH_NO_DEDUCED_FUNCTION_POINTERS)
   funcp = boost::math::expm1<value_type>;
#else
   funcp = boost::math::expm1;
#endif
   result = boost::math::tools::test_hetero<Real>(
      data, 
      bind_func<Real>(funcp, 0), 
      extract_result<Real>(2));
   handle_test_result(result, data[result.worst()], result.worst(), type_name, "expm1", "Random test data");
   std::cout << std::endl;
#endif
}

template <class T>
void test(T, const char* type_name)
{
#  include "log1p_expm1_data.ipp"

   do_test<T>(log1p_expm1_data, type_name, "expm1 and log1p");

   //
   // C99 Appendix F special cases:
   static const T zero = 0;
   static const T m_one = -1;
   BOOST_CHECK_EQUAL(boost::math::log1p(zero), zero);
   BOOST_CHECK_EQUAL(boost::math::log1p(T(-zero)), zero);
   BOOST_CHECK_EQUAL(boost::math::expm1(zero), zero);
   if(std::numeric_limits<T>::has_infinity)
   {
      BOOST_CHECK_EQUAL(boost::math::log1p(m_one), -std::numeric_limits<T>::infinity());
      BOOST_CHECK_EQUAL(boost::math::expm1(T(-std::numeric_limits<T>::infinity())), m_one);
      BOOST_CHECK_EQUAL(boost::math::expm1(std::numeric_limits<T>::infinity()), std::numeric_limits<T>::infinity());
#ifndef BOOST_BORLANDC
#ifndef BOOST_NO_EXCEPTIONS
      // When building with Borland's compiler, simply the *presence*
      // of these tests cause other unrelated tests to fail!!! :-(
      using namespace boost::math::policies;
      typedef policy<overflow_error<throw_on_error> > pol;
      BOOST_MATH_CHECK_THROW(boost::math::log1p(m_one, pol()), std::overflow_error);
      BOOST_MATH_CHECK_THROW(boost::math::expm1(std::numeric_limits<T>::infinity(), pol()), std::overflow_error);
#endif
#endif
   }
}

