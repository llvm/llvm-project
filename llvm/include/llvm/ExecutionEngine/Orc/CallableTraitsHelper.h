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

#ifndef LLVM_EXECUTIONENGINE_ORC_CALLABLETRAITSHELPER_H
#define LLVM_EXECUTIONENGINE_ORC_CALLABLETRAITSHELPER_H

#include <tuple>
#include <type_traits>

namespace llvm::orc {

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
  using ReturnType = RetT;
  using ArgsTupleType = std::tuple<ArgTs...>;
};
} // namespace detail

/// CallableArgInfo provides typedefs for the return type and argument types
/// (as a tuple) of the given callable type.
template <typename Callable>
struct CallableArgInfo
    : public CallableTraitsHelper<detail::CallableArgInfoImpl, Callable> {};

} // namespace llvm::orc

#endif // LLVM_EXECUTIONENGINE_ORC_CALLABLETRAITSHELPER_H
