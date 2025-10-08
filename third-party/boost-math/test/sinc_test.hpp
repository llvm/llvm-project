// unit test file sinc.hpp for the special functions test suite

//  (C) Copyright Hubert Holin 2003.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <functional>
#include <iomanip>
#include <iostream>
#include <complex>


#include <boost/math/special_functions/sinc.hpp>
#include <boost/multiprecision/cpp_complex.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>


BOOST_TEST_CASE_TEMPLATE_FUNCTION(sinc_pi_test, T)
{
    using    ::std::abs;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::sinc_pi;
    
    
    BOOST_TEST_MESSAGE("Testing sinc_pi in the real domain for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(sinc_pi<T>(static_cast<T>(0))-static_cast<T>(1)))
        (numeric_limits<T>::epsilon()));
}


BOOST_TEST_CASE_TEMPLATE_FUNCTION(sinc_pi_complex_test, T)
{
    using    ::std::abs;
    using    ::std::sinh;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::sinc_pi;
    
    
    BOOST_TEST_MESSAGE("Testing sinc_pi in the complex domain for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(sinc_pi<T>(::std::complex<T>(0, 1))-
             ::std::complex<T>(sinh(static_cast<T>(1)))))
        (numeric_limits<T>::epsilon()));

    //
    // A very poor test with rather large tolerance for failure.
    // But it does get out coverage up!
    //
    BOOST_MATH_IF_CONSTEXPR(std::numeric_limits<T>::is_specialized && std::numeric_limits<T>::digits < 60)
    {
       T tolerance = std::numeric_limits<T>::epsilon() * 20000;

       std::complex<T> val(1, 2);
       for (unsigned i = 0; i < 5; ++i)
       {
          using mp_t = boost::multiprecision::cpp_complex_100;
          val /= 3;
          std::complex<T> r1, r2;
          r1 = sinc_pi<T>(val);
          r2 = static_cast<std::complex<T>>(sin(mp_t(val)) / mp_t(val));
          BOOST_CHECK_CLOSE_FRACTION(arg(r1), arg(r2), tolerance);
          BOOST_CHECK_CLOSE_FRACTION(abs(r1), abs(r2), tolerance);
       }
    }
}


void    sinc_pi_manual_check()
{
    using    ::boost::math::sinc_pi;
    
    
    BOOST_TEST_MESSAGE(" ");
    BOOST_TEST_MESSAGE("sinc_pi");
    
    for    (int i = 0; i <= 100; i++)
    {
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
        BOOST_TEST_MESSAGE( ::std::setw(15)
                    << sinc_pi<float>(static_cast<float>(i-50)/
                                                static_cast<float>(50))
                    << ::std::setw(15)
                    << sinc_pi<double>(static_cast<double>(i-50)/
                                                static_cast<double>(50))
                    << ::std::setw(15)
                    << sinc_pi<long double>(static_cast<long double>(i-50)/
                                                static_cast<long double>(50)));
#else
        BOOST_TEST_MESSAGE( ::std::setw(15)
                    << sinc_pi<float>(static_cast<float>(i-50)/
                                                static_cast<float>(50))
                    << ::std::setw(15)
                    << sinc_pi<double>(static_cast<double>(i-50)/
                                                static_cast<double>(50)));
#endif
    }
    
    BOOST_TEST_MESSAGE(" ");
}


