//===-- move_only_function.h - moveable, type-erasing function --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Provides orc_rt::move_only_function, a substitute for
/// std::move_only_function usable prior to C++23. See the
/// move_only_function class documentation for supported signatures and
/// semantics.
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

#include "orc-rt/CallableTraitsHelper.h"

#include <memory>
#include <type_traits>

namespace orc_rt {

namespace move_only_function_detail {

/// Type-erased call interface. Two partial specializations on IsConst
/// select a const or non-const virtual call operator, so an
/// InvocableStorage's operator() can dispatch through the vtable with
/// the correct cv-qualifier.
template <bool IsConst, bool IsNoexcept, typename RetT, typename... ArgTs>
class Callable;

template <bool IsNoexcept, typename RetT, typename... ArgTs>
class Callable<false, IsNoexcept, RetT, ArgTs...> {
public:
  virtual ~Callable() = default;
  virtual RetT call(ArgTs &&...Args) noexcept(IsNoexcept) = 0;
};

template <bool IsNoexcept, typename RetT, typename... ArgTs>
class Callable<true, IsNoexcept, RetT, ArgTs...> {
public:
  virtual ~Callable() = default;
  virtual RetT call(ArgTs &&...Args) const noexcept(IsNoexcept) = 0;
};

/// Concrete Callable holding a callable object and forwarding
/// invocations to it. The templated constructor static-asserts that the
/// stored callable is nothrow-invocable when IsNoexcept is true.
template <typename CallableT, bool IsConst, bool IsNoexcept, typename RetT,
          typename... ArgTs>
class CallableImpl;

template <typename CallableT, bool IsNoexcept, typename RetT, typename... ArgTs>
class CallableImpl<CallableT, false, IsNoexcept, RetT, ArgTs...>
    : public Callable<false, IsNoexcept, RetT, ArgTs...> {
public:
  template <typename CallableInitT>
  CallableImpl(CallableInitT &&Callable)
      : Callable(std::forward<CallableInitT>(Callable)) {
    static_assert(
        !IsNoexcept ||
            std::is_nothrow_invocable_r_v<RetT, CallableT &, ArgTs...>,
        "Callable stored in a noexcept-qualified move_only_function must be "
        "nothrow-invocable.");
  }
  RetT call(ArgTs &&...Args) noexcept(IsNoexcept) override {
    return Callable(std::forward<ArgTs>(Args)...);
  }

private:
  CallableT Callable;
};

template <typename CallableT, bool IsNoexcept, typename RetT, typename... ArgTs>
class CallableImpl<CallableT, true, IsNoexcept, RetT, ArgTs...>
    : public Callable<true, IsNoexcept, RetT, ArgTs...> {
public:
  template <typename CallableInitT>
  CallableImpl(CallableInitT &&Callable)
      : Callable(std::forward<CallableInitT>(Callable)) {
    static_assert(
        !IsNoexcept ||
            std::is_nothrow_invocable_r_v<RetT, const CallableT &, ArgTs...>,
        "Callable stored in a const noexcept-qualified move_only_function "
        "must be nothrow-invocable.");
  }
  RetT call(ArgTs &&...Args) const noexcept(IsNoexcept) override {
    return Callable(std::forward<ArgTs>(Args)...);
  }

private:
  CallableT Callable;
};

/// Owns the type-erased Callable pointer and exposes state-only ops
/// (operator bool). Shared base for both InvocableStorage specializations.
template <bool IsConst, bool IsNoexcept, typename RetT, typename... ArgTs>
class Storage {
protected:
  template <typename CallableT>
  using WrappedCallable =
      CallableImpl<CallableT, IsConst, IsNoexcept, RetT, ArgTs...>;

  std::unique_ptr<Callable<IsConst, IsNoexcept, RetT, ArgTs...>> C;

public:
  explicit operator bool() const noexcept { return !!C; }
};

/// Extends Storage with an operator() qualified per IsConst and
/// IsNoexcept. Two specializations on IsConst yield either a non-const
/// or const operator().
template <bool IsConst, bool IsNoexcept, typename RetT, typename... ArgTs>
class InvocableStorage;

template <bool IsNoexcept, typename RetT, typename... ArgTs>
class InvocableStorage<false, IsNoexcept, RetT, ArgTs...>
    : public Storage<false, IsNoexcept, RetT, ArgTs...> {
public:
  RetT operator()(ArgTs... Params) noexcept(IsNoexcept) {
    return this->C->call(std::forward<ArgTs>(Params)...);
  }
};

template <bool IsNoexcept, typename RetT, typename... ArgTs>
class InvocableStorage<true, IsNoexcept, RetT, ArgTs...>
    : public Storage<true, IsNoexcept, RetT, ArgTs...> {
public:
  RetT operator()(ArgTs... Params) const noexcept(IsNoexcept) {
    return this->C->call(std::forward<ArgTs>(Params)...);
  }
};

/// Applies CallableTraitsHelper to a signature FnT to decompose it into
/// (IsConst, IsNoexcept, RetT, ArgTs...) and expose the resulting
/// InvocableStorage as the base of move_only_function.
template <typename FnT>
class MOFBase : public CallableTraitsHelper<InvocableStorage, FnT> {};

} // namespace move_only_function_detail

/// A move-only, type-erasing wrapper for a callable. Mirrors C++23's
/// std::move_only_function.
///
/// The template argument FnT is a function type that may carry the const
/// and/or noexcept cv-qualifiers:
///
///     R(A...)                - mutable, may throw
///     R(A...) const          - const-callable, may throw
///     R(A...) noexcept       - mutable, guaranteed nothrow
///     R(A...) const noexcept - const-callable, guaranteed nothrow
///
/// operator()'s const and noexcept qualifiers are derived from the
/// signature; constructing a noexcept-qualified move_only_function from a
/// throwing callable is rejected at compile time (via a static_assert on
/// nothrow-invocability), and constructing a const-qualified
/// move_only_function from a mutable callable is rejected at compile time
/// (via the const-qualified call to the stored callable).
template <typename FnT>
class move_only_function : public move_only_function_detail::MOFBase<FnT> {
public:
  move_only_function() = default;
  move_only_function(std::nullptr_t) {}
  move_only_function(move_only_function &&) = default;
  move_only_function(const move_only_function &) = delete;
  move_only_function &operator=(move_only_function &&) = default;
  move_only_function &operator=(const move_only_function &) = delete;

  template <typename CallableT> move_only_function(CallableT &&C) {
    using WrappedCallable = typename move_only_function_detail::MOFBase<
        FnT>::template WrappedCallable<CallableT>;
    this->C = std::make_unique<WrappedCallable>(std::forward<CallableT>(C));
  }
};

} // namespace orc_rt

#endif // ORC_RT_MOVE_ONLY_FUNCTION_H
