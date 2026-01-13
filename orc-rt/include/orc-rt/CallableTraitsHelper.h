//===- CallableTraitsHelper.h - Callable arg/ret type extractor -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CallableTraitsHelper API.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_CALLABLETRAITSHELPER_H
#define ORC_RT_CALLABLETRAITSHELPER_H

#include <tuple>
#include <type_traits>

namespace orc_rt {

/// CallableTraitsHelper takes an implementation class template Impl and some
/// callable type C and passes the return and argument types of C to the Impl
/// class template.
///
/// This can be used to simplify the implementation of classes that need to
/// operate on callable types.
template <template <typename...> typename ImplT, typename C>
struct CallableTraitsHelper
    : public CallableTraitsHelper<
          ImplT,
          decltype(&std::remove_cv_t<std::remove_reference_t<C>>::operator())> {
};

template <template <typename...> typename ImplT, typename RetT,
          typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT(ArgTs...)>
    : public ImplT<RetT, ArgTs...> {};

template <template <typename...> typename ImplT, typename RetT,
          typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (*)(ArgTs...)>
    : public CallableTraitsHelper<ImplT, RetT(ArgTs...)> {};

template <template <typename...> typename ImplT, typename RetT,
          typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (&)(ArgTs...)>
    : public CallableTraitsHelper<ImplT, RetT(ArgTs...)> {};

template <template <typename...> typename ImplT, typename ClassT, typename RetT,
          typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (ClassT::*)(ArgTs...)>
    : public CallableTraitsHelper<ImplT, RetT(ArgTs...)> {};

template <template <typename...> typename ImplT, typename ClassT, typename RetT,
          typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (ClassT::*)(ArgTs...) const>
    : public CallableTraitsHelper<ImplT, RetT(ArgTs...)> {};

namespace detail {
template <typename RetT, typename... ArgTs> struct CallableArgInfoImpl {
  typedef RetT return_type;
  typedef std::tuple<ArgTs...> args_tuple_type;
};
} // namespace detail

/// CallableArgInfo provides typedefs for the return type and argument types
/// (as a tuple) of the given callable type.
template <typename Callable>
struct CallableArgInfo
    : public CallableTraitsHelper<detail::CallableArgInfoImpl, Callable> {};

} // namespace orc_rt

#endif // ORC_RT_CALLABLETRAITSHELPER_H
