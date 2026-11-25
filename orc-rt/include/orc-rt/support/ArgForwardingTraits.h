//===-- ArgForwardingTraits.h - Forward deserialized args -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// ArgForwardingTraits API.
//
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_SUPPORT_ARGFORWARDINGTRAITS_H
#define ORC_RT_SUPPORT_ARGFORWARDINGTRAITS_H

#include "orc-rt/support/CallableTraitsHelper.h"

#include <tuple>
#include <type_traits>
#include <utility>

namespace orc_rt::detail {

/// Utilities for calling a handler with arguments that have been deserialized
/// into a tuple. ArgTs should be the handler's declared parameter types.
template <typename... ArgTs> struct ArgForwardingTraits {
  /// The type to deserialize arguments into: the handler's parameter types,
  /// decayed.
  typedef std::tuple<std::decay_t<ArgTs>...> ArgTupleType;

  // Forwards arguments based on the parameter types of the handler.
  //
  // The call operator takes lvalue references to the deserialized
  // (ArgTupleType) values and forwards each according to the handler's
  // declared parameter type: T and T&& parameters receive rvalues, while
  // const T& and T& parameters receive lvalues. Modifications made through a
  // T& parameter are made to the deserialized value. The handler's return
  // value (if any) is returned.
  template <typename FnT> class ForwardArgsAsRequested {
  public:
    ForwardArgsAsRequested(FnT Fn) : Fn(std::move(Fn)) {}
    decltype(auto) operator()(ArgTs &...Args) {
      return Fn(std::forward<ArgTs>(Args)...);
    }

  private:
    FnT Fn;
  };

  template <typename FnT>
  static ForwardArgsAsRequested<std::decay_t<FnT>>
  forwardArgsAsRequested(FnT &&Fn) {
    return ForwardArgsAsRequested<std::decay_t<FnT>>(std::forward<FnT>(Fn));
  }
};

template <bool /* is_const */, bool /* is_noexcept */, typename RetT,
          typename... ArgTs>
struct ArgForwardingTraitsAdapter : ArgForwardingTraits<ArgTs...> {};

/// ArgForwardingTraits for the parameter types of callable type C.
template <typename C>
using ArgForwardingTraitsFor =
    CallableTraitsHelper<ArgForwardingTraitsAdapter, C>;

} // namespace orc_rt::detail

#endif // ORC_RT_SUPPORT_ARGFORWARDINGTRAITS_H
