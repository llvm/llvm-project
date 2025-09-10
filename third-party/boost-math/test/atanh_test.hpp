// unit test file atanh.hpp for the special functions test suite

//  (C) Copyright Hubert Holin 2003.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <functional>
#include <iomanip>
//#include <iostream>

#define BOOST_TEST_MAiN
#include <boost/math/special_functions/atanh.hpp>


#include <boost/test/unit_test.hpp>

template<typename T>
T    atanh_error_evaluator(T x)
{
    using    ::std::abs;
    using    ::std::tanh;
    using    ::std::cosh;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::atanh;
    
    
    static T const   epsilon = numeric_limits<float>::epsilon();
    
    T                y = tanh(x);
    T                z = atanh(y);
    
    T                absolute_error = abs(z-x);
    T                relative_error = absolute_error/(cosh(x)*cosh(x));
    T                scaled_error = relative_error/epsilon;
    
    return(scaled_error);
}


BOOST_TEST_CASE_TEMPLATE_FUNCTION(atanh_test, T)
{
    using    ::std::abs;
    using    ::std::tanh;
    using    ::std::log;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::atanh;
    
    
    BOOST_TEST_MESSAGE("Testing atanh in the real domain for "
        << string_type_name<T>::_() << ".");
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(atanh<T>(static_cast<T>(0))))
        (numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(atanh<T>(static_cast<T>(3)/5) - log(static_cast<T>(2))))
        (numeric_limits<T>::epsilon()));
    
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
        (abs(atanh<T>(static_cast<T>(-3)/5) + log(static_cast<T>(2))))
        (numeric_limits<T>::epsilon()));
    
    for    (int i = 0; i <= 100; i++)
    {
        T    x = static_cast<T>(i-50)/static_cast<T>(5);
        T    y = tanh(x);
        
        if    (
                (abs(y-static_cast<T>(1)) >= numeric_limits<T>::epsilon())&&
                (abs(y+static_cast<T>(1)) >= numeric_limits<T>::epsilon())
            )
        {
            BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
                (atanh_error_evaluator(x))
                (static_cast<T>(4)));
        }
    }
    //
    // Error handling checks:
    //
    BOOST_MATH_CHECK_THROW(atanh(T(-1)), std::overflow_error);
    BOOST_MATH_CHECK_THROW(atanh(T(1)), std::overflow_error);
    BOOST_MATH_CHECK_THROW(atanh(T(-2)), std::domain_error);
    BOOST_MATH_CHECK_THROW(atanh(T(2)), std::domain_error);
    if (std::numeric_limits<T>::has_quiet_NaN)
    {
       T n = std::numeric_limits<T>::quiet_NaN();
       BOOST_CHECK_THROW(boost::math::atanh(n), std::domain_error);
    }
}


void    atanh_manual_check()
{
    using    ::std::abs;
    using    ::std::tanh;
        
    using    ::std::numeric_limits;
    
    
    BOOST_TEST_MESSAGE(" ");
    BOOST_TEST_MESSAGE("atanh");
    
    for    (int i = 0; i <= 100; i++)
    {
        float        xf = static_cast<float>(i-50)/static_cast<float>(5);
        double       xd = static_cast<double>(i-50)/static_cast<double>(5);
        long double  xl =
                static_cast<long double>(i-50)/static_cast<long double>(5);
        
        float        yf = tanh(xf);
        double       yd = tanh(xd);
        (void) &yd;        // avoid "unused variable" warning
        long double  yl = tanh(xl);
        (void) &yl;        // avoid "unused variable" warning
        
        if    (
                std::numeric_limits<float>::has_infinity &&
                std::numeric_limits<double>::has_infinity &&
                std::numeric_limits<long double>::has_infinity
            )
        {
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
            BOOST_TEST_MESSAGE( ::std::setw(15)
                        << atanh_error_evaluator(xf)
                        << ::std::setw(15)
                        << atanh_error_evaluator(xd)
                        << ::std::setw(15)
                        << atanh_error_evaluator(xl));
#else
            BOOST_TEST_MESSAGE( ::std::setw(15)
                        << atanh_error_evaluator(xf)
                        << ::std::setw(15)
                        << atanh_error_evaluator(xd));
#endif
        }
        else
        {
            if    (
                    (abs(yf-static_cast<float>(1)) <
                        numeric_limits<float>::epsilon())||
                    (abs(yf+static_cast<float>(1)) <
                        numeric_limits<float>::epsilon())||
                    (abs(yf-static_cast<double>(1)) <
                        numeric_limits<double>::epsilon())||
                    (abs(yf+static_cast<double>(1)) <
                        numeric_limits<double>::epsilon())||
                    (abs(yf-static_cast<long double>(1)) <
                        numeric_limits<long double>::epsilon())||
                    (abs(yf+static_cast<long double>(1)) <
                        numeric_limits<long double>::epsilon())
                )
            {
                BOOST_TEST_MESSAGE("Platform's numerics may lack precision.");
            }
            else
            {
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
                BOOST_TEST_MESSAGE( ::std::setw(15)
                            << atanh_error_evaluator(xf)
                            << ::std::setw(15)
                            << atanh_error_evaluator(xd)
                            << ::std::setw(15)
                            << atanh_error_evaluator(xl));
#else
                BOOST_TEST_MESSAGE( ::std::setw(15)
                            << atanh_error_evaluator(xf)
                            << ::std::setw(15)
                            << atanh_error_evaluator(xd));
#endif
            }
        }
    }
    
    BOOST_TEST_MESSAGE(" ");
}

