// unit test file sinhc.hpp for the special functions test suite

//  (C) Copyright Hubert Holin 2003.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <functional>
#include <iomanip>
#include <iostream>
#include <complex>


#include <boost/math/special_functions/sinhc.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>


BOOST_TEST_CASE_TEMPLATE_FUNCTION(sinhc_pi_test, T)
{
    using    ::std::abs;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::sinhc_pi;
    
    
    BOOST_TEST_MESSAGE("Testing sinhc_pi in the real domain for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(sinhc_pi<T>(static_cast<T>(0))-static_cast<T>(1)))
        (numeric_limits<T>::epsilon()));
}


BOOST_TEST_CASE_TEMPLATE_FUNCTION(sinhc_pi_complex_test, T)
{
    using    ::std::abs;
    using    ::std::sin;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::sinhc_pi;
    
    
    BOOST_TEST_MESSAGE("Testing sinhc_pi in the complex domain for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(sinhc_pi<T>(::std::complex<T>(0, 1))-
             ::std::complex<T>(sin(static_cast<T>(1)))))
        (numeric_limits<T>::epsilon()));
}


void    sinhc_pi_manual_check()
{
    using    ::boost::math::sinhc_pi;
    
    
    BOOST_TEST_MESSAGE(" ");
    BOOST_TEST_MESSAGE("sinc_pi");
    
    for    (int i = 0; i <= 100; i++)
    {
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
        BOOST_TEST_MESSAGE( ::std::setw(15)
                    << sinhc_pi<float>(static_cast<float>(i-50)/
                                                static_cast<float>(50))
                    << ::std::setw(15)
                    << sinhc_pi<double>(static_cast<double>(i-50)/
                                                static_cast<double>(50))
                    << ::std::setw(15)
                    << sinhc_pi<long double>(static_cast<long double>(i-50)/
                                                static_cast<long double>(50)));
#else
        BOOST_TEST_MESSAGE( ::std::setw(15)
                    << sinhc_pi<float>(static_cast<float>(i-50)/
                                                static_cast<float>(50))
                    << ::std::setw(15)
                    << sinhc_pi<double>(static_cast<double>(i-50)/
                                                static_cast<double>(50)));
#endif
    }
    
    BOOST_TEST_MESSAGE(" ");
}
    

