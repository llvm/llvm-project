// unit test file asinh.hpp for the special functions test suite

//  (C) Copyright Hubert Holin 2003.
//  Distributed under the Boost Software License, Version 1.0. (See
//  accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)


#include <functional>
#include <iomanip>
#include <iostream>

#define BOOST_TEST_MAiN
#include <boost/math/special_functions/asinh.hpp>


#include <boost/test/unit_test.hpp>


template<typename T>
T    asinh_error_evaluator(T x)
{
    using    ::std::abs;
    using    ::std::sinh;
    using    ::std::cosh;
        
    using    ::std::numeric_limits;
    
    using    ::boost::math::asinh;
    
    
    static T const    epsilon = numeric_limits<float>::epsilon();
    
    T                y = sinh(x);
    T                z = asinh(y);
    
    T                absolute_error = abs(z-x);
    T                relative_error = absolute_error*cosh(x);
    T                scaled_error = relative_error/epsilon;
    
    return(scaled_error);
}


BOOST_TEST_CASE_TEMPLATE_FUNCTION(asinh_test, T)
{
    BOOST_TEST_MESSAGE("Testing asinh in the real domain for "
        << string_type_name<T>::_() << ".");
    
    for    (int i = 0; i <= 80; i++)
    {
        T    x = static_cast<T>(i-40)/static_cast<T>(4);
        
        BOOST_CHECK_PREDICATE(::std::less_equal<T>(),
            (asinh_error_evaluator(x))
            (static_cast<T>(4)));
    }
    //
    // Special cases:
    //
    if(std::numeric_limits<T>::has_infinity)
    {
       T inf = std::numeric_limits<T>::infinity();
       boost::math::policies::policy<boost::math::policies::overflow_error<boost::math::policies::ignore_error> > pol;
       BOOST_CHECK_EQUAL(boost::math::asinh(inf, pol), inf);
       BOOST_CHECK_EQUAL(boost::math::asinh(-inf, pol), -inf);
    }
    if(std::numeric_limits<T>::has_quiet_NaN)
    {
       T n = std::numeric_limits<T>::quiet_NaN();
       BOOST_CHECK_THROW(boost::math::asinh(n), std::domain_error);
    }
}


void    asinh_manual_check()
{
    BOOST_TEST_MESSAGE(" ");
    BOOST_TEST_MESSAGE("asinh");
    
    for    (int i = 0; i <= 80; i++)
    {
        float        xf = static_cast<float>(i-40)/static_cast<float>(4);
        double       xd = static_cast<double>(i-40)/static_cast<double>(4);
        long double  xl =
                static_cast<long double>(i-40)/static_cast<long double>(4);
        
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
        BOOST_TEST_MESSAGE(  ::std::setw(15)
                     << asinh_error_evaluator(xf)
                     << ::std::setw(15)
                     << asinh_error_evaluator(xd)
                     << ::std::setw(15)
                     << asinh_error_evaluator(xl));
#else
        BOOST_TEST_MESSAGE(  ::std::setw(15)
                     << asinh_error_evaluator(xf)
                     << ::std::setw(15)
                     << asinh_error_evaluator(xd));
#endif
    }
    
    BOOST_TEST_MESSAGE(" ");
}

