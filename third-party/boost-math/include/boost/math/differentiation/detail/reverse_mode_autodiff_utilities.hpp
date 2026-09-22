//           Copyright Maksym Zhelyeznyakov 2025-2026.
// Distributed under the Boost Software License, Version 1.0.
//      (See accompanying file LICENSE_1_0.txt or copy at
//           https://www.boost.org/LICENSE_1_0.txt)
#ifndef REVERSE_MODE_AUTODIFF_UTILITIES_HPP
#define REVERSE_MODE_AUTODIFF_UTILITIES_HPP

#include <utility>

namespace boost {
namespace math {
namespace differentiation {
namespace reverse_mode {
namespace detail {

template<bool Condition>
struct if_functional_dispatch_impl;

template<>
struct if_functional_dispatch_impl<true>
{
    template<typename F1, typename F2, typename... Args>
    static decltype(auto) call(F1&& f1, F2&&, Args&&... args)
    {
        return std::forward<F1>(f1)(std::forward<Args>(args)...);
    }
};

template<>
struct if_functional_dispatch_impl<false>
{
    template<typename F1, typename F2, typename... Args>
    static decltype(auto) call(F1&&, F2&& f2, Args&&... args)
    {
        return std::forward<F2>(f2)(std::forward<Args>(args)...);
    }
};

template<bool Condition, typename F1, typename F2, typename... Args>
decltype(auto) if_functional_dispatch(F1&& f1, F2&& f2, Args&&... args)
{
    return if_functional_dispatch_impl<Condition>::call(std::forward<F1>(f1),
                                                        std::forward<F2>(f2),
                                                        std::forward<Args>(args)...);
}
} // namespace detail
} // namespace reverse_mode
} // namespace differentiation
} // namespace math
} // namespace boost
#endif // REVERSE_MODE_AUTODIFF_UTILITIES_HPP
