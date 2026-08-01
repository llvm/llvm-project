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
template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename C>
struct CallableTraitsHelper
    : CallableTraitsHelper<
          ImplT,
          decltype(&std::remove_cv_t<std::remove_reference_t<C>>::operator())> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT(ArgTs...)>
    : ImplT</* is_const = */ false, /* is_noexcept = */ false, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT(ArgTs...) noexcept>
    : ImplT</* is_const = */ false, /* is_noexcept = */ true, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT(ArgTs...) const>
    : ImplT</* is_const = */ true, /* is_noexcept = */ false, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT(ArgTs...) const noexcept>
    : ImplT</* is_const = */ true, /* is_noexcept = */ true, RetT, ArgTs...> {};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (*)(ArgTs...)>
    : ImplT</* is_const = */ false, /* is_noexcept = */ false, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (*)(ArgTs...) noexcept>
    : ImplT</* is_const = */ false, /* is_noexcept = */ true, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (&)(ArgTs...)>
    : ImplT</* is_const = */ false, /* is_noexcept = */ false, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (&)(ArgTs...) noexcept>
    : ImplT</* is_const = */ false, /* is_noexcept = */ true, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename ClassT, typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (ClassT::*)(ArgTs...)>
    : ImplT</* is_const = */ false, /* is_noexcept = */ false, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename ClassT, typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (ClassT::*)(ArgTs...) noexcept>
    : ImplT</* is_const = */ false, /* is_noexcept = */ true, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename ClassT, typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (ClassT::*)(ArgTs...) const>
    : ImplT</* is_const = */ true, /* is_noexcept = */ false, RetT, ArgTs...> {
};

template <template <bool /* is_const */, bool /* is_noexcept */,
                    typename...> typename ImplT,
          typename ClassT, typename RetT, typename... ArgTs>
struct CallableTraitsHelper<ImplT, RetT (ClassT::*)(ArgTs...) const noexcept>
    : ImplT</* is_const = */ true, /* is_noexcept = */ true, RetT, ArgTs...> {};

namespace detail {
template <bool IsConst, bool IsNoexcept, typename RetT, typename... ArgTs>
struct CallableArgInfoImpl {
  static constexpr bool is_const = IsConst;
  static constexpr bool is_noexcept = IsNoexcept;
  typedef RetT return_type;
  typedef std::tuple<ArgTs...> args_tuple_type;
};
} // namespace detail

/// CallableArgInfo provides typedefs for the return type and argument types
/// (as a tuple) of the given callable type.
template <typename Callable>
struct CallableArgInfo
    : CallableTraitsHelper<detail::CallableArgInfoImpl, Callable> {};

} // namespace orc_rt

#endif // ORC_RT_CALLABLETRAITSHELPER_H
