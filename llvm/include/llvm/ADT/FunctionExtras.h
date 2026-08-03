//===- FunctionExtras.h - Function type erasure utilities -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file provides a collection of function (or more generally, callable)
/// type erasure utilities supplementing those provided by the standard library
/// in `<function>`.
///
/// It provides `unique_function`, which works like `std::function` but supports
/// move-only callable objects and const-qualification.
///
/// Future plans:
/// - Add a `function` that provides ref-qualified support, which doesn't work
///   with `std::function`.
/// - Provide support for specifying multiple signatures to type erase callable
///   objects with an overload set, such as those produced by generic lambdas.
/// - Expand to include a copyable utility that directly replaces std::function
///   but brings the above improvements.
///
/// Note that LLVM's utilities are greatly simplified by not supporting
/// allocators.
///
/// If the standard library ever begins to provide comparable facilities we can
/// consider switching to those.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_FUNCTIONEXTRAS_H
#define LLVM_ADT_FUNCTIONEXTRAS_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/MemAlloc.h"
#include "llvm/Support/type_traits.h"
#include <cstring>
#include <type_traits>

namespace llvm {

/// unique_function is a type-erasing functor similar to std::function.
///
/// It can hold move-only function objects, like lambdas capturing unique_ptrs.
/// Accordingly, it is movable but not copyable.
///
/// It supports const-qualification:
/// - unique_function<int() const> has a const operator().
///   It can only hold functions which themselves have a const operator().
/// - unique_function<int()> has a non-const operator().
///   It can hold functions with a non-const operator(), like mutable lambdas.
template <typename FunctionT> class unique_function;

namespace detail {

template <typename CallableT, typename ThisT>
using EnableUnlessSameType =
    std::enable_if_t<!std::is_same<remove_cvref_t<CallableT>, ThisT>::value>;
template <typename CallableT, typename Ret, typename... Params>
using EnableIfCallable = std::enable_if_t<std::disjunction<
    std::is_void<Ret>,
    std::is_same<decltype(std::declval<CallableT>()(std::declval<Params>()...)),
                 Ret>,
    std::is_same<const decltype(std::declval<CallableT>()(
                     std::declval<Params>()...)),
                 Ret>,
    std::is_convertible<decltype(std::declval<CallableT>()(
                            std::declval<Params>()...)),
                        Ret>>::value>;

template <typename ReturnT, typename... ParamTs> class UniqueFunctionBase {
protected:
  static constexpr size_t InlineStorageSize = sizeof(void *) * 3;
  static constexpr size_t InlineStorageAlign = alignof(void *);

  // Provide a type function to map parameters that won't observe extra copies
  // or moves and which are small enough to likely pass in register to values
  // and all other types to l-value reference types. We use this to compute the
  // types used in our erased call utility to minimize copies and moves unless
  // doing so would force things unnecessarily into memory.
  //
  // The heuristic used is related to common ABI register passing conventions.
  // It doesn't have to be exact though, and in one way it is more strict
  // because we want to still be able to observe either moves *or* copies.
  template <typename T> struct AdjustedParamTBase {
    static_assert(!std::is_reference<T>::value,
                  "references should be handled by template specialization");
    static constexpr bool IsSizeLessThanThreshold =
        sizeof(T) <= 2 * sizeof(void *);
    using type =
        std::conditional_t<std::is_trivially_copy_constructible<T>::value &&
                               std::is_trivially_move_constructible<T>::value &&
                               IsSizeLessThanThreshold,
                           T, T &>;
  };

  // This specialization ensures that 'AdjustedParam<V<T>&>' or
  // 'AdjustedParam<V<T>&&>' does not trigger a compile-time error when 'T' is
  // an incomplete type and V a templated type.
  template <typename T> struct AdjustedParamTBase<T &> { using type = T &; };
  template <typename T> struct AdjustedParamTBase<T &&> { using type = T &; };

  template <typename T>
  using AdjustedParamT = typename AdjustedParamTBase<T>::type;

  // The type of the erased function pointer we use as a callback to dispatch to
  // the stored callable when it is trivial to move and destroy.
  using CallPtrT = ReturnT (*)(const UniqueFunctionBase *Self,
                               AdjustedParamT<ParamTs>... Params);
  using DestroyMovePtrT = void (*)(UniqueFunctionBase *LHS,
                                   UniqueFunctionBase *RHS);

  // The main storage buffer. This will either have a pointer to out-of-line
  // storage or an inline buffer storing the callable.
  union StorageT {
    // For out-of-line storage we keep a pointer to the underlying storage.
    void *OutOfLine;
    static_assert(
        sizeof(OutOfLine) <= InlineStorageSize,
        "Should always use all of the out-of-line storage for inline storage!");

    // For in-line storage, we just provide an aligned character buffer. We
    // provide three pointers worth of storage here.
    // This is mutable as an inlined `const unique_function<void() const>` may
    // still modify its own mutable members.
    alignas(InlineStorageAlign) mutable std::byte Inline[InlineStorageSize];
  } Storage;

  CallPtrT CallPtr = nullptr;
  DestroyMovePtrT DestroyMovePtr = nullptr;

  // A simple tag type so the call-as type to be passed to the constructor.
  template <typename T> struct CalledAs {};

  // Essentially the "main" unique_function constructor, but subclasses
  // provide the qualified type to be used for the call.
  // (We always store a T, even if the call will use a pointer to const T).
  template <typename CallableT, typename CalledAsT>
  UniqueFunctionBase(CallableT Callable, CalledAs<CalledAsT>) {
    // static as workaround an MSVC bug which treats constexpr uses as odr-uses.
    static constexpr bool UsesInlineStorage =
        sizeof(CallableT) <= InlineStorageSize &&
        alignof(CallableT) <= InlineStorageAlign;
    void *CallableAddr = &Storage.Inline;
    if constexpr (!UsesInlineStorage) {
      CallableAddr = allocate_buffer(sizeof(CallableT), alignof(CallableT));
      Storage.OutOfLine = CallableAddr;
    }

    // Now move into the storage.
    new (CallableAddr) CallableT(std::move(Callable));

    CallPtr = [](const UniqueFunctionBase *Self,
                 AdjustedParamT<ParamTs>... Params) -> ReturnT {
      void *CallableAddr =
          UsesInlineStorage ? &Self->Storage.Inline : Self->Storage.OutOfLine;
      auto &Func = *reinterpret_cast<CalledAsT *>(CallableAddr);
      return Func(std::forward<ParamTs>(Params)...);
    };

    // For trivial inline storage, we don't need to do anything on move/destroy.
    if constexpr (!std::is_trivially_move_constructible_v<CallableT> ||
                  !std::is_trivially_destructible_v<CallableT> ||
                  !UsesInlineStorage) {
      // If LHS is set, move LHS callable to RHS, then destroy RHS callable.
      DestroyMovePtr = [](UniqueFunctionBase *LHS, UniqueFunctionBase *RHS) {
        if constexpr (!UsesInlineStorage) {
          if (LHS) {
            // Out-of-line move: just move the pointer.
            LHS->Storage.OutOfLine = RHS->Storage.OutOfLine;
          } else {
            // Out-of-line destroy.
            void *RHSCallableAddr = RHS->Storage.OutOfLine;
            reinterpret_cast<CallableT *>(RHSCallableAddr)->~CallableT();
            deallocate_buffer(RHSCallableAddr, sizeof(CallableT),
                              alignof(CallableT));
          }
        } else {
          auto *RHSCallable =
              reinterpret_cast<CallableT *>(&RHS->Storage.Inline);
          if (LHS) {
            // Inline move: move-construct first...
            auto *LHSCallable =
                reinterpret_cast<CallableT *>(&LHS->Storage.Inline);
            new (LHSCallable) CallableT(std::move(*RHSCallable));
          }
          // ... destroy RHS in any case.
          RHSCallable->~CallableT();
        }
      };
    }
  }

  ~UniqueFunctionBase() {
    if (DestroyMovePtr)
      DestroyMovePtr(nullptr, this);
  }

  UniqueFunctionBase(UniqueFunctionBase &&RHS) noexcept {
    CallPtr = RHS.CallPtr;
    DestroyMovePtr = RHS.DestroyMovePtr;
    if (DestroyMovePtr)
      DestroyMovePtr(this, &RHS);
    else // Trivial callable stored inline => memcpy.
      memcpy(&Storage.Inline, &RHS.Storage.Inline, InlineStorageSize);

    RHS.CallPtr = nullptr;
    RHS.DestroyMovePtr = nullptr; // Moved everything out of RHS.
#ifndef NDEBUG
    // In debug builds, we also scribble across the rest of the storage.
    memset(RHS.Storage.Inline, 0xAD, InlineStorageSize);
#endif
  }

  UniqueFunctionBase &operator=(UniqueFunctionBase &&RHS) noexcept {
    if (this == &RHS)
      return *this;

    // Because we don't try to provide any exception safety guarantees we can
    // implement move assignment very simply by first destroying the current
    // object and then move-constructing over top of it.
    this->~UniqueFunctionBase();
    new (this) UniqueFunctionBase(std::move(RHS));
    return *this;
  }

  UniqueFunctionBase() = default;

public:
  explicit operator bool() const { return (bool)CallPtr; }
};

} // namespace detail

template <typename R, typename... P>
class unique_function<R(P...)> : public detail::UniqueFunctionBase<R, P...> {
  using Base = detail::UniqueFunctionBase<R, P...>;

public:
  unique_function() = default;
  unique_function(std::nullptr_t) {}
  unique_function(unique_function &&) = default;
  unique_function(const unique_function &) = delete;
  unique_function &operator=(unique_function &&) = default;
  unique_function &operator=(const unique_function &) = delete;

  template <typename CallableT>
  unique_function(
      CallableT Callable,
      detail::EnableUnlessSameType<CallableT, unique_function> * = nullptr,
      detail::EnableIfCallable<CallableT, R, P...> * = nullptr)
      : Base(std::forward<CallableT>(Callable),
             typename Base::template CalledAs<CallableT>{}) {}

  R operator()(P... Params) { return this->CallPtr(this, Params...); }
};

template <typename R, typename... P>
class unique_function<R(P...) const>
    : public detail::UniqueFunctionBase<R, P...> {
  using Base = detail::UniqueFunctionBase<R, P...>;

public:
  unique_function() = default;
  unique_function(std::nullptr_t) {}
  unique_function(unique_function &&) = default;
  unique_function(const unique_function &) = delete;
  unique_function &operator=(unique_function &&) = default;
  unique_function &operator=(const unique_function &) = delete;

  template <typename CallableT>
  unique_function(
      CallableT Callable,
      detail::EnableUnlessSameType<CallableT, unique_function> * = nullptr,
      detail::EnableIfCallable<const CallableT, R, P...> * = nullptr)
      : Base(std::forward<CallableT>(Callable),
             typename Base::template CalledAs<const CallableT>{}) {}

  R operator()(P... Params) const { return this->CallPtr(this, Params...); }
};

} // end namespace llvm

#endif // LLVM_ADT_FUNCTIONEXTRAS_H
