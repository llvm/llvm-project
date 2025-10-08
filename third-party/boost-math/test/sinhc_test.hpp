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
#include <boost/multiprecision/cpp_bin_float.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>


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

    using mp_t = boost::multiprecision::cpp_bin_float_50;
    T val = 2;
    T tolerance = boost::math::tools::epsilon<T>() * 100;
    for (unsigned i = 0; i < 1000; ++i)
    {
       val /= 3;
       if (val < boost::math::tools::min_value<T>())
          break;
       T r1 = sinhc_pi(val);
       T r2 = static_cast<T>(sinh(mp_t(val)) / mp_t(val));
       BOOST_CHECK_CLOSE_FRACTION(r1, r2, tolerance);
    }
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

    using mp_t = boost::multiprecision::cpp_complex_50;
    std::complex<T> val(2, 2.5);
    for (unsigned i = 0; i < 50; ++i)
    {
       val /= 3;
       if (val.real() < boost::math::tools::min_value<T>())
          break;
       std::complex<T> r1 = sinhc_pi(val);
       std::complex<T> r2 = static_cast<std::complex<T>>(sinh(mp_t(val)) / mp_t(val));
       BOOST_CHECK_LE(std::abs(r1.real() - r2.real()), boost::math::tools::epsilon<T>());
       BOOST_CHECK_LE(std::abs(r1.imag() - r2.imag()), boost::math::tools::epsilon<T>());
    }
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
    

