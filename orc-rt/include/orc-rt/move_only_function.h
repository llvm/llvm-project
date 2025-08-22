//===-- move_only_function.h - moveable, type-erasing function --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// A substitute for std::move_only_function that can be used until the ORC
/// runtime is allowed to assume c++-23.
///
/// TODO: Replace all uses with std::move_only_function once we can assume
///       c++-23.
///
/// TODO: Re-implement using techniques from LLVM's unique_function
///       (llvm/include/llvm/ADT/FunctionExtras.h), which uses some extra
///       inline storage to avoid heap allocations for small objects. This
///       would require first porting some other LLVM utilities like
///       PointerIntPair, PointerUnion, and PointerLikeTypeTraits. (These are
///       likely to be independently useful in the orc runtime, so porting will
///       have additional benefits).
///
//===----------------------------------------------------------------------===//

#ifndef ORC_RT_MOVE_ONLY_FUNCTION_H
#define ORC_RT_MOVE_ONLY_FUNCTION_H

#include <memory>

namespace orc_rt {

namespace move_only_function_detail {

template <typename RetT, typename... ArgTs> class Callable {
public:
  virtual ~Callable() = default;
  virtual RetT call(ArgTs &&...Args) = 0;
};

template <typename CallableT, typename RetT, typename... ArgTs>
class CallableImpl : public Callable<RetT, ArgTs...> {
public:
  CallableImpl(CallableT &&Callable) : Callable(std::move(Callable)) {}
  RetT call(ArgTs &&...Args) override {
    return Callable(std::forward<ArgTs>(Args)...);
  }

private:
  CallableT Callable;
};

} // namespace move_only_function_detail

template <typename FnT> class move_only_function;

template <typename RetT, typename... ArgTs>
class move_only_function<RetT(ArgTs...)> {
public:
  move_only_function() = default;
  move_only_function(std::nullptr_t) {}
  move_only_function(move_only_function &&) = default;
  move_only_function(const move_only_function &&) = delete;
  move_only_function &operator=(move_only_function &&) = default;
  move_only_function &operator=(const move_only_function &&) = delete;

  template <typename CallableT>
  move_only_function(CallableT &&Callable)
      : C(std::make_unique<
            move_only_function_detail::CallableImpl<CallableT, RetT, ArgTs...>>(
            std::forward<CallableT>(Callable))) {}

  RetT operator()(ArgTs... Params) {
    return C->call(std::forward<ArgTs>(Params)...);
  }

  explicit operator bool() const { return !!C; }

private:
  std::unique_ptr<move_only_function_detail::Callable<RetT, ArgTs...>> C;
};

} // namespace orc_rt

#endif // ORC_RT_MOVE_ONLY_FUNCTION_H
