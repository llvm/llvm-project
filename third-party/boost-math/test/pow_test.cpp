//  Boost pow_test.cpp test file
//  Tests the pow function

//  (C) Copyright Bruno Lalande 2008.
//  (C) Copyright Matt Borland 2024.
//  Distributed under the Boost Software License, Version 1.0.
//  (See accompanying file LICENSE_1_0.txt or copy at
//  http://www.boost.org/LICENSE_1_0.txt)

#include <cmath>
#include <string>
#include <iostream>

#include <boost/math/concepts/real_concept.hpp>
#include "../include_private/boost/math/tools/test.hpp"
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <boost/typeof/typeof.hpp>
#include <boost/math/tools/assert.hpp>

#include <boost/math/special_functions/pow.hpp>

#include BOOST_TYPEOF_INCREMENT_REGISTRATION_GROUP()
BOOST_TYPEOF_REGISTER_TYPE(boost::math::concepts::real_concept)

using namespace boost;
using namespace boost::math;

template <int N, class T>
void test_pow(T base)
{
    typedef typename tools::promote_args<T>::type result_type;

    BOOST_MATH_STD_USING

    if ((base == 0) && N < 0)
    {
       #ifndef BOOST_MATH_NO_EXCEPTIONS 
       BOOST_MATH_CHECK_THROW(math::pow<N>(base), std::overflow_error);
       #endif
    }
    else
    {
       BOOST_CHECK_CLOSE(math::pow<N>(base),
              pow(static_cast<result_type>(base), static_cast<result_type>(N)),
              boost::math::tools::epsilon<result_type>() * 100 * 400); // 400 eps as a %
    }
}

template <int N, class T>
void test_with_big_bases()
{
    for (T base = T(); base < T(1000); ++base)
        test_pow<N>(base);
}

template <int N, class T>
void test_with_small_bases()
{
    T base = 0.9f;
    for (int i = 0; i < 10; ++i)
    {
        base += base/50;
        test_pow<N>(base);
    }
}

template <class T, int Factor>
void test_with_small_exponents()
{
    test_with_big_bases<0, T>();
    test_with_big_bases<Factor*1, T>();
    test_with_big_bases<Factor*2, T>();
    test_with_big_bases<Factor*3, T>();
    test_with_big_bases<Factor*5, T>();
    test_with_big_bases<Factor*6, T>();
    test_with_big_bases<Factor*7, T>();
    test_with_big_bases<Factor*8, T>();
    test_with_big_bases<Factor*9, T>();
    test_with_big_bases<Factor*10, T>();
    test_with_big_bases<Factor*11, T>();
    test_with_big_bases<Factor*12, T>();
}

template <class T, int Factor>
void test_with_big_exponents()
{
    test_with_small_bases<Factor*50, T>();
    test_with_small_bases<Factor*100, T>();
    test_with_small_bases<Factor*150, T>();
    test_with_small_bases<Factor*200, T>();
    test_with_small_bases<Factor*250, T>();
    test_with_small_bases<Factor*300, T>();
    test_with_small_bases<Factor*350, T>();
    test_with_small_bases<Factor*400, T>();
    test_with_small_bases<Factor*450, T>();
    test_with_small_bases<Factor*500, T>();
}


void test_return_types()
{
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>('\1')), double>::value), "Return type mismatch");
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>(L'\2')), double>::value), "Return type mismatch");
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>(3)), double>::value), "Return type mismatch");
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>(4u)), double>::value), "Return type mismatch");
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>(5ul)), double>::value), "Return type mismatch");
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>(6.0f)), float>::value), "Return type mismatch");
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>(7.0)), double>::value), "Return type mismatch");
#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    static_assert((boost::math::is_same<BOOST_TYPEOF(pow<2>(7.0l)), long double>::value), "Return type mismatch");
#endif
}


namespace boost { namespace math { namespace policies {
template <class T>
T user_overflow_error(const char*, const char*, const T&)
{ return T(123.45); }
}}}

namespace boost { namespace math { namespace policies {
template <class T>
T user_indeterminate_result_error(const char*, const char*, const T&)
{ return T(456.78); }
}}}


void test_error_policy()
{
    using namespace policies;

    BOOST_CHECK(pow<-2>(
                    0.0,
                    policy< ::boost::math::policies::overflow_error<user_error> >()
                )
                == 123.45);

    BOOST_CHECK(pow<0>(
                    0.0,
                    policy< ::boost::math::policies::indeterminate_result_error<user_error> >()
                )
                == 456.78);
}

BOOST_AUTO_TEST_CASE( test_main )
{
    using namespace std;

    cout << "Testing with integral bases and positive small exponents" << endl;
    test_with_small_exponents<int, 1>();
    cout << "Testing with integral bases and negative small exponents" << endl;
    test_with_small_exponents<int, -1>();

    cout << "Testing with float precision bases and positive small exponents" << endl;
    test_with_small_exponents<float, 1>();
    cout << "Testing with float precision bases and negative small exponents" << endl;
    test_with_small_exponents<float, -1>();

    cout << "Testing with float precision bases and positive big exponents" << endl;
    test_with_big_exponents<float, 1>();
    cout << "Testing with float precision bases and negative big exponents" << endl;
    test_with_big_exponents<float, -1>();

     cout << "Testing with double precision bases and positive small exponents" << endl;
    test_with_small_exponents<double, 1>();
    cout << "Testing with double precision bases and negative small exponents" << endl;
    test_with_small_exponents<double, -1>();

    cout << "Testing with double precision bases and positive big exponents" << endl;
    test_with_big_exponents<double, 1>();
    cout << "Testing with double precision bases and negative big exponents" << endl;
    test_with_big_exponents<double, -1>();

#ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    cout << "Testing with long double precision bases and positive small exponents" << endl;
    test_with_small_exponents<long double, 1>();
    cout << "Testing with long double precision bases and negative small exponents" << endl;
    test_with_small_exponents<long double, -1>();

    cout << "Testing with long double precision bases and positive big exponents" << endl;
    test_with_big_exponents<long double, 1>();
    cout << "Testing with long double precision bases and negative big exponents" << endl;
    test_with_big_exponents<long double, -1>();

    cout << "Testing with concepts::real_concept precision bases and positive small exponents" << endl;
    test_with_small_exponents<boost::math::concepts::real_concept, 1>();
    cout << "Testing with concepts::real_concept precision bases and negative small exponents" << endl;
    test_with_small_exponents<boost::math::concepts::real_concept, -1>();

    cout << "Testing with concepts::real_concept precision bases and positive big exponents" << endl;
    test_with_big_exponents<boost::math::concepts::real_concept, 1>();
    cout << "Testing with concepts::real_concept precision bases and negative big exponents" << endl;
    test_with_big_exponents<boost::math::concepts::real_concept, -1>();
#endif

    #ifndef BOOST_NO_CXX14_CONSTEXPR
    static_assert(boost::math::pow<8>(2)==256, "Pow is not constexpr");
    static_assert(boost::math::pow<9>(2)==512, "Pow is not constexpr");
    static_assert(boost::math::pow<2>(2)==4,   "Pow is not constexpr");
    static_assert(boost::math::pow<1>(2)==2,   "Pow is not constexpr");
    static_assert(boost::math::pow<0>(2)==1,   "Pow is not constexpr");
    #endif

    test_return_types();

    test_error_policy();
}

/*

  Running 1 test case...
  Testing with integral bases and positive small exponents
  Testing with integral bases and negative small exponents
  Testing with float precision bases and positive small exponents
  Testing with float precision bases and negative small exponents
  Testing with float precision bases and positive big exponents
  Testing with float precision bases and negative big exponents
  Testing with double precision bases and positive small exponents
  Testing with double precision bases and negative small exponents
  Testing with double precision bases and positive big exponents
  Testing with double precision bases and negative big exponents
  Testing with long double precision bases and positive small exponents
  Testing with long double precision bases and negative small exponents
  Testing with long double precision bases and positive big exponents
  Testing with long double precision bases and negative big exponents
  Testing with concepts::real_concept precision bases and positive small exponents
  Testing with concepts::real_concept precision bases and negative small exponents
  Testing with concepts::real_concept precision bases and positive big exponents
  Testing with concepts::real_concept precision bases and negative big exponents
  
  *** No errors detected

  */
