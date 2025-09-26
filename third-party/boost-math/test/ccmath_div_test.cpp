//  (C) Copyright Matt Borland 2021.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <cstdint>
#include <cinttypes>
#include <boost/math/ccmath/div.hpp>
#include <boost/math/tools/is_constant_evaluated.hpp>
#include <boost/multiprecision/cpp_int.hpp>

template <typename Z>
constexpr void test()
{
    constexpr auto test_val1 = boost::math::ccmath::div(Z(1), Z(1));
    static_assert(test_val1.quot == Z(1));
    static_assert(test_val1.rem == Z(0));

    constexpr auto test_val2 = boost::math::ccmath::div(Z(1'000'000), Z(3));
    static_assert(test_val2.quot == Z(333'333));
    static_assert(test_val2.rem == Z(1));
}

#if !defined(BOOST_MATH_NO_CONSTEXPR_DETECTION) && !defined(BOOST_MATH_USING_BUILTIN_CONSTANT_P)
int main()
{
    test<int>();
    test<long>();
    test<long long>();
    test<std::intmax_t>();

    test<std::int32_t>();
    test<std::int64_t>();
    test<std::uint32_t>();

    test<boost::multiprecision::int128_t>();
    test<boost::multiprecision::int256_t>();
    test<boost::multiprecision::int512_t>();

    return 0;
}
#else
int main()
{
    return 0;
}
#endif
