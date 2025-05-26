//  Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#define BOOST_MATH_STANDALONE
#define BOOST_MP_STANDALONE

#include <boost/config.hpp>
#include <boost/math/distributions/uniform.hpp>
#include <boost/multiprecision/cpp_bin_float.hpp>
#include <type_traits>

int main()
{
    using T = boost::multiprecision::cpp_bin_float_quad;
    
    // This macro should be available through MP standalone since it bundles config
    BOOST_IF_CONSTEXPR (std::is_same<T, boost::multiprecision::cpp_bin_float_quad>::value)
    {
        boost::math::uniform_distribution<boost::multiprecision::cpp_bin_float_quad> d {0, 1};
        const auto q = boost::math::quantile(d, T(0.5));
        BOOST_MATH_ASSERT(q == T(0.5));
    }

    return 0;
}
