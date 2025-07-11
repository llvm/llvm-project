// unit test file acosh.hpp for the special functions test suite

//  (C) Copyright Hubert Holin 2003.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <functional>
#include <iomanip>
#include <iostream>

#define BOOST_TEST_MAiN
#include <boost/math/special_functions/acosh.hpp>


#include <boost/test/unit_test.hpp>


template<typename T>
T    acosh_error_evaluator(T x)
{
    using    ::std::abs;
    using    ::std::sinh;
    using    ::std::cosh;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::acosh;
    
    
    static T const    epsilon = numeric_limits<float>::epsilon();
    
    T                y = cosh(x);
    T                z = acosh(y);
    
    T                absolute_error = abs(z-abs(x));
    T                relative_error = absolute_error*abs(sinh(x));
    T                scaled_error = relative_error/epsilon;
    
    return(scaled_error);
}


BOOST_TEST_CASE_TEMPLATE_FUNCTION(acosh_test, T)
{
    BOOST_TEST_MESSAGE("Testing acosh in the real domain for "
        << string_type_name<T>::_() << ".");
    
    for    (int i = 0; i <= 100; i++)
    {
        T    x = static_cast<T>(i-50)/static_cast<T>(5);
        
        BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
            (acosh_error_evaluator(x))
            (static_cast<T>(4)));
    }
    // special cases for bug report: https://svn.boost.org/trac/boost/ticket/5113
    T x = 1e-2f;
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
       (acosh_error_evaluator(x))
       (static_cast<T>(4)));
    x = 1e-3f;
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
       (acosh_error_evaluator(x))
       (static_cast<T>(4)));
    x = 1e-4f;
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
       (acosh_error_evaluator(x))
       (static_cast<T>(4)));
    x = 1e-5f;
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
       (acosh_error_evaluator(x))
       (static_cast<T>(4)));
    x = 1e-6f;
    BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
       (acosh_error_evaluator(x))
       (static_cast<T>(4)));
    //
    // Special cases:
    //
    if(std::numeric_limits<T>::has_infinity)
    {
       T inf = std::numeric_limits<T>::infinity();
       boost::math::policies::policy<boost::math::policies::overflow_error<boost::math::policies::ignore_error> > pol;
       BOOST_CHECK_EQUAL(boost::math::acosh(inf, pol), inf);
    }
    x = -1;
    BOOST_CHECK_THROW(boost::math::acosh(x), std::domain_error);
    if (std::numeric_limits<T>::has_quiet_NaN)
    {
       T n = std::numeric_limits<T>::quiet_NaN();
       BOOST_CHECK_THROW(boost::math::acosh(n), std::domain_error);
    }
}


void    acosh_manual_check()
{
    BOOST_TEST_MESSAGE(" ");
    BOOST_TEST_MESSAGE("acosh");
    
    for    (int i = 0; i <= 100; i++)
    {
        float        xf = static_cast<float>(i-50)/static_cast<float>(5);
        double       xd = static_cast<double>(i-50)/static_cast<double>(5);
        long double  xl = 
                static_cast<long double>(i-50)/static_cast<long double>(5);
        
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
        BOOST_TEST_MESSAGE(  ::std::setw(15)
                     << acosh_error_evaluator(xf)
                     << ::std::setw(15)
                     << acosh_error_evaluator(xd)
                     << ::std::setw(15)
                     << acosh_error_evaluator(xl));
#else
        BOOST_TEST_MESSAGE(  ::std::setw(15)
                     << acosh_error_evaluator(xf)
                     << ::std::setw(15)
                     << acosh_error_evaluator(xd));
#endif
    }
    
    BOOST_TEST_MESSAGE(" ");
}

