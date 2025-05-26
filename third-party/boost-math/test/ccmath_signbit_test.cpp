//  (C) Copyright Matt Borland 2022.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <type_traits>
#include <boost/math/ccmath/signbit.hpp>

#if !defined(BOOST_MATH_NO_CONSTEXPR_DETECTION) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P)
template <typename T>
void test()
{
    // Edge cases
    #ifdef BOOST_MATH_BIT_CAST
    static_assert(boost::math::ccmath::signbit(T(0)) == false);
    static_assert(boost::math::ccmath::signbit(T(0)*-1) == true);

    static_assert(boost::math::ccmath::signbit(std::numeric_limits<T>::quiet_NaN()) == false);
    static_assert(boost::math::ccmath::signbit(-std::numeric_limits<T>::quiet_NaN()) == true);

    static_assert(boost::math::ccmath::signbit(std::numeric_limits<T>::signaling_NaN()) == false);
    static_assert(boost::math::ccmath::signbit(-std::numeric_limits<T>::signaling_NaN()) == true);
    #endif

    // Positive numbers
    static_assert(boost::math::ccmath::signbit(std::numeric_limits<T>::infinity()) == false);
    static_assert(boost::math::ccmath::signbit(T(1)) == false);

    // Negative numbers
    static_assert(boost::math::ccmath::signbit(-std::numeric_limits<T>::infinity()) == true);
    static_assert(boost::math::ccmath::signbit(T(-1)) == true);
}

int main(void)
{
    test<float>();
    test<double>();

    #ifndef BOOST_MATH_NO_LONG_DOUBLE_MATH_FUNCTIONS
    test<long double>();
    #endif

    return 0;
}
#else
int main(void)
{
    return 0;
}
#endif 
