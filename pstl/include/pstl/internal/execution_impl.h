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

template <typename _IteratorTag, typename... _IteratorTypes>
auto
__is_iterator_of(int) -> decltype(
    std::conjunction<std::is_base_of<
        _IteratorTag, typename std::iterator_traits<std::decay_t<_IteratorTypes>>::iterator_category>...>{});

template <typename... _IteratorTypes>
auto
__is_iterator_of(...) -> std::false_type;

template <typename... _IteratorTypes>
struct __is_random_access_iterator : decltype(__is_iterator_of<std::random_access_iterator_tag, _IteratorTypes...>(0))
{
};

template <typename... _IteratorTypes>
struct __is_forward_iterator : decltype(__is_iterator_of<std::forward_iterator_tag, _IteratorTypes...>(0))
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

struct __serial_backend
{
};
struct __tbb_backend
{
};

using __par_backend_tag =
#ifdef _PSTL_PAR_BACKEND_TBB
    __tbb_backend;
#elif _PSTL_PAR_BACKEND_SERIAL
    __serial_backend;
#else
#    error "A parallel backend must be specified";
#endif

template <class _IsVector>
struct __serial_tag
{
    using __is_vector = _IsVector;
};

template <class _IsVector>
struct __parallel_tag
{
    using __is_vector = _IsVector;
    // backend tag can be change depending on
    // TBB availability in the environment
    using __backend_tag = __par_backend_tag;
};

struct __parallel_forward_tag
{
    using __is_vector = std::false_type;
    // backend tag can be change depending on
    // TBB availability in the environment
    using __backend_tag = __par_backend_tag;
};

template <class _IsVector, class... _IteratorTypes>
using __tag_type =
    typename std::conditional<__internal::__is_random_access_iterator<_IteratorTypes...>::value,
                              __parallel_tag<_IsVector>,
                              typename std::conditional<__is_forward_iterator<_IteratorTypes...>::value,
                                                        __parallel_forward_tag, __serial_tag<_IsVector>>::type>::type;

template <class... _IteratorTypes>
__serial_tag</*_IsVector = */ std::false_type>
__select_backend(std::execution::sequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__serial_tag<__internal::__is_random_access_iterator<_IteratorTypes...>>
__select_backend(std::execution::unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type</*_IsVector = */ std::false_type, _IteratorTypes...>
__select_backend(std::execution::parallel_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type<__internal::__is_random_access_iterator<_IteratorTypes...>, _IteratorTypes...>
__select_backend(std::execution::parallel_unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

} // namespace __internal
} // namespace __pstl

_PSTL_HIDE_FROM_ABI_POP

#endif /* _PSTL_EXECUTION_IMPL_H */
