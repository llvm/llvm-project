//  (C) Copyright John Maddock 2010.
//  (C) Copyright Matt Borland 2024.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef BOOST_MATH_TUPLE_HPP_INCLUDED
#define BOOST_MATH_TUPLE_HPP_INCLUDED

#include <boost/math/tools/config.hpp>

#ifdef BOOST_MATH_ENABLE_CUDA

#include <boost/math/tools/type_traits.hpp>
#include <cuda/std/utility>
#include <cuda/std/tuple>

namespace boost { 
namespace math {

using cuda::std::pair;
using cuda::std::tuple;

using cuda::std::make_pair;

using cuda::std::tie;
using cuda::std::get;

using cuda::std::tuple_size;
using cuda::std::tuple_element;

namespace detail {

template <typename T>
BOOST_MATH_GPU_ENABLED T&& forward(boost::math::remove_reference_t<T>& arg) noexcept
{
    return static_cast<T&&>(arg);
}

template <typename T>
BOOST_MATH_GPU_ENABLED T&& forward(boost::math::remove_reference_t<T>&& arg) noexcept
{
    static_assert(!boost::math::is_lvalue_reference<T>::value, "Cannot forward an rvalue as an lvalue.");
    return static_cast<T&&>(arg);
}

} // namespace detail

template <typename T, typename... Ts>
BOOST_MATH_GPU_ENABLED auto make_tuple(T&& t, Ts&&... ts) 
{
    return cuda::std::tuple<boost::math::decay_t<T>, boost::math::decay_t<Ts>...>(
        boost::math::detail::forward<T>(t), boost::math::detail::forward<Ts>(ts)...
    );
}

} // namespace math
} // namespace boost

#else

#include <tuple>

namespace boost { 
namespace math {

using ::std::tuple;
using ::std::pair;

// [6.1.3.2] Tuple creation functions
using ::std::ignore;
using ::std::make_tuple;
using ::std::tie;
using ::std::get;

// [6.1.3.3] Tuple helper classes
using ::std::tuple_size;
using ::std::tuple_element;

// Pair helpers
using ::std::make_pair;

} // namespace math
} // namespace boost

#endif // BOOST_MATH_ENABLE_CUDA

#endif
