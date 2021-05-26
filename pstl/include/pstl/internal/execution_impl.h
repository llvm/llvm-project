// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _PSTL_EXECUTION_IMPL_H
#define _PSTL_EXECUTION_IMPL_H

#include <iterator>
#include <type_traits>

#include "pstl_config.h"
#include "execution_defs.h"

_PSTL_HIDE_FROM_ABI_PUSH

namespace __pstl
{
namespace __internal
{

using namespace __pstl::execution;

template <typename _IteratorType>
struct __is_random_access_iterator
    : std::is_same<typename std::iterator_traits<_IteratorType>::iterator_category,
                   std::random_access_iterator_tag>
{
};

template <typename Policy>
struct __policy_traits
{
};

template <>
struct __policy_traits<sequenced_policy>
{
    typedef std::false_type __allow_parallel;
    typedef std::false_type __allow_unsequenced;
    typedef std::false_type __allow_vector;
};

template <>
struct __policy_traits<unsequenced_policy>
{
    typedef std::false_type __allow_parallel;
    typedef std::true_type __allow_unsequenced;
    typedef std::true_type __allow_vector;
};

template <>
struct __policy_traits<parallel_policy>
{
    typedef std::true_type __allow_parallel;
    typedef std::false_type __allow_unsequenced;
    typedef std::false_type __allow_vector;
};

template <>
struct __policy_traits<parallel_unsequenced_policy>
{
    typedef std::true_type __allow_parallel;
    typedef std::true_type __allow_unsequenced;
    typedef std::true_type __allow_vector;
};

template <typename _ExecutionPolicy>
using __allow_vector =
    typename __internal::__policy_traits<typename std::decay<_ExecutionPolicy>::type>::__allow_vector;

template <typename _ExecutionPolicy>
using __allow_unsequenced =
    typename __internal::__policy_traits<typename std::decay<_ExecutionPolicy>::type>::__allow_unsequenced;

template <typename _ExecutionPolicy>
using __allow_parallel =
    typename __internal::__policy_traits<typename std::decay<_ExecutionPolicy>::type>::__allow_parallel;

template <typename _ExecutionPolicy, typename... _IteratorTypes>
typename std::conjunction<__allow_vector<_ExecutionPolicy>,
                          __is_random_access_iterator<_IteratorTypes>...>::type
__is_vectorization_preferred(_ExecutionPolicy&&)
{
    return {};
}

template <typename _ExecutionPolicy, typename... _IteratorTypes>
typename std::conjunction<__allow_parallel<_ExecutionPolicy>,
                          __is_random_access_iterator<_IteratorTypes>...>::type
__is_parallelization_preferred(_ExecutionPolicy&&)
{
    return {};
}

} // namespace __internal
} // namespace __pstl

_PSTL_HIDE_FROM_ABI_POP

#endif /* _PSTL_EXECUTION_IMPL_H */
