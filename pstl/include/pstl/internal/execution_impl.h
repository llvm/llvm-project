// -*- C++ -*-
//===-- execution_impl.h --------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef __PSTL_execution_impl_H
#define __PSTL_execution_impl_H

#include <iterator>
#include <type_traits>

#include "execution_defs.h"

namespace __pstl
{
namespace internal
{

using namespace __pstl::execution;

/* predicate */

template <typename _Tp>
std::false_type lazy_and(_Tp, std::false_type)
{
    return std::false_type{};
};

template <typename _Tp>
inline _Tp
lazy_and(_Tp __a, std::true_type)
{
    return __a;
}

template <typename _Tp>
std::true_type lazy_or(_Tp, std::true_type)
{
    return std::true_type{};
};

template <typename _Tp>
inline _Tp
lazy_or(_Tp __a, std::false_type)
{
    return __a;
}

/* iterator */
template <typename _IteratorType, typename... _OtherIteratorTypes>
struct is_random_access_iterator
{
    static constexpr bool value =
        is_random_access_iterator<_IteratorType>::value && is_random_access_iterator<_OtherIteratorTypes...>::value;
    typedef std::integral_constant<bool, value> type;
};

template <typename _IteratorType>
struct is_random_access_iterator<_IteratorType>
    : std::is_same<typename std::iterator_traits<_IteratorType>::iterator_category, std::random_access_iterator_tag>
{
};

/* policy */
template <typename Policy>
struct policy_traits
{
};

template <>
struct policy_traits<sequenced_policy>
{
    typedef std::false_type allow_parallel;
    typedef std::false_type allow_unsequenced;
    typedef std::false_type allow_vector;
};

template <>
struct policy_traits<unsequenced_policy>
{
    typedef std::false_type allow_parallel;
    typedef std::true_type allow_unsequenced;
    typedef std::true_type allow_vector;
};

#if __PSTL_USE_PAR_POLICIES
template <>
struct policy_traits<parallel_policy>
{
    typedef std::true_type allow_parallel;
    typedef std::false_type allow_unsequenced;
    typedef std::false_type allow_vector;
};

template <>
struct policy_traits<parallel_unsequenced_policy>
{
    typedef std::true_type allow_parallel;
    typedef std::true_type allow_unsequenced;
    typedef std::true_type allow_vector;
};
#endif

template <typename _ExecutionPolicy>
using collector_t = typename policy_traits<typename std::decay<_ExecutionPolicy>::type>::collector_type;

template <typename _ExecutionPolicy>
using allow_vector = typename internal::policy_traits<typename std::decay<_ExecutionPolicy>::type>::allow_vector;

template <typename _ExecutionPolicy>
using allow_unsequenced =
    typename internal::policy_traits<typename std::decay<_ExecutionPolicy>::type>::allow_unsequenced;

template <typename _ExecutionPolicy>
using allow_parallel = typename internal::policy_traits<typename std::decay<_ExecutionPolicy>::type>::allow_parallel;

template <typename _ExecutionPolicy, typename... _IteratorTypes>
auto
is_vectorization_preferred(_ExecutionPolicy&& __exec)
    -> decltype(lazy_and(__exec.__allow_vector(), typename is_random_access_iterator<_IteratorTypes...>::type()))
{
    return internal::lazy_and(__exec.__allow_vector(), typename is_random_access_iterator<_IteratorTypes...>::type());
}

template <typename _ExecutionPolicy, typename... _IteratorTypes>
auto
is_parallelization_preferred(_ExecutionPolicy&& __exec)
    -> decltype(lazy_and(__exec.__allow_parallel(), typename is_random_access_iterator<_IteratorTypes...>::type()))
{
    return internal::lazy_and(__exec.__allow_parallel(), typename is_random_access_iterator<_IteratorTypes...>::type());
}

template <typename policy, typename... _IteratorTypes>
struct prefer_unsequenced_tag
{
    static constexpr bool value =
        allow_unsequenced<policy>::value && is_random_access_iterator<_IteratorTypes...>::value;
    typedef std::integral_constant<bool, value> type;
};

template <typename policy, typename... _IteratorTypes>
struct prefer_parallel_tag
{
    static constexpr bool value = allow_parallel<policy>::value && is_random_access_iterator<_IteratorTypes...>::value;
    typedef std::integral_constant<bool, value> type;
};

} // namespace internal
} // namespace __pstl

#endif /* __PSTL_execution_impl_H */
