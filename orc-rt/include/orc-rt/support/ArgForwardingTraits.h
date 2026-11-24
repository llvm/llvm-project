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
  template <typename FnT> class ForwardArgsAsRequested {
  public:
    ForwardArgsAsRequested(FnT &&Fn) : Fn(std::move(Fn)) {}
    void operator()(ArgTs &...Args) { Fn(std::forward<ArgTs>(Args)...); }

  private:
    FnT Fn;
  };

  template <typename FnT>
  static ForwardArgsAsRequested<std::decay_t<FnT>>
  forwardArgsAsRequested(FnT &&Fn) {
    return ForwardArgsAsRequested<std::decay_t<FnT>>(std::forward<FnT>(Fn));
  }
};

} // namespace orc_rt::detail

#endif // ORC_RT_SUPPORT_ARGFORWARDINGTRAITS_H
