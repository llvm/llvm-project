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

template <typename _IteratorTag, typename... _IteratorTypes>
using __are_iterators_of = std::conjunction<
    std::is_base_of<_IteratorTag, typename std::iterator_traits<std::decay_t<_IteratorTypes>>::iterator_category>...>;

template <typename... _IteratorTypes>
using __are_random_access_iterators = __are_iterators_of<std::random_access_iterator_tag, _IteratorTypes...>;

struct __serial_backend_tag
{
};
struct __tbb_backend_tag
{
};
struct __openmp_backend_tag
{
};

#if defined(_PSTL_PAR_BACKEND_TBB)
using __par_backend_tag = __tbb_backend_tag;
#elif defined(_PSTL_PAR_BACKEND_OPENMP)
using __par_backend_tag = __openmp_backend_tag;
#elif defined(_PSTL_PAR_BACKEND_SERIAL)
using __par_backend_tag = __serial_backend_tag;
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

template <class _IsVector, class... _IteratorTypes>
using __tag_type = typename std::conditional<__internal::__are_random_access_iterators<_IteratorTypes...>::value,
                                             __parallel_tag<_IsVector>, __serial_tag<_IsVector>>::type;

template <class... _IteratorTypes>
__serial_tag</*_IsVector = */ std::false_type>
__select_backend(__pstl::execution::sequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__serial_tag<__internal::__are_random_access_iterators<_IteratorTypes...>>
__select_backend(__pstl::execution::unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type</*_IsVector = */ std::false_type, _IteratorTypes...>
__select_backend(__pstl::execution::parallel_policy, _IteratorTypes&&...)
{
    return {};
}

template <class... _IteratorTypes>
__tag_type<__internal::__are_random_access_iterators<_IteratorTypes...>, _IteratorTypes...>
__select_backend(__pstl::execution::parallel_unsequenced_policy, _IteratorTypes&&...)
{
    return {};
}

} // namespace __internal
} // namespace __pstl

_PSTL_HIDE_FROM_ABI_POP

#endif /* _PSTL_EXECUTION_IMPL_H */
