//===- llvm/Analysis/WithCache.h - KnownBits cache for pointers -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Store a pointer to any type along with the KnownBits information for it
// that is computed lazily (if required).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_WITHCACHE_H
#define LLVM_ANALYSIS_WITHCACHE_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/KnownBits.h"
#include <type_traits>

namespace llvm {
struct SimplifyQuery;
LLVM_ABI KnownBits computeKnownBits(const Value *V, const SimplifyQuery &Q,
                                    unsigned Depth);

template <typename Arg, typename = void> class WithCache {
  static_assert(std::is_pointer_v<Arg>, "WithCache requires a pointer type!");

  using UnderlyingType = std::remove_pointer_t<Arg>;
  constexpr static bool IsConst = std::is_const_v<Arg>;

  template <typename T, bool Const>
  using conditionally_const_t = std::conditional_t<Const, const T, T>;

  using PointerType = conditionally_const_t<UnderlyingType *, IsConst>;
  using ReferenceType = conditionally_const_t<UnderlyingType &, IsConst>;

  // Store the presence of the KnownBits information in one of the bits of
  // Pointer.
  // true  -> present
  // false -> absent
  mutable PointerIntPair<PointerType, 1, bool> Pointer;
  mutable KnownBits Known;

  void calculateKnownBits(const SimplifyQuery &Q) const {
    Known = computeKnownBits(Pointer.getPointer(), Q, 0);
    Pointer.setInt(true);
  }

public:
  WithCache(PointerType Pointer) : Pointer(Pointer, false) {}
  WithCache(PointerType Pointer, const KnownBits &Known)
      : Pointer(Pointer, true), Known(Known) {}

  /// Prevent slicing for the derived class, to avoid perf foot-guns around
  /// duplicated caches.
  template <typename U,
            typename = std::enable_if_t<!std::is_const_v<U> &&
                                        std::is_same_v<const U *, Arg>>>
  WithCache(const WithCache<U *> &) = delete;

  /// Prevent slicing for the derived class, to avoid perf foot-guns around
  /// duplicated caches.
  template <typename U,
            typename = std::enable_if_t<!std::is_const_v<U> &&
                                        std::is_same_v<const U *, Arg>>>
  WithCache &operator=(const WithCache<U *> &) = delete;

  [[nodiscard]] PointerType getValue() const { return Pointer.getPointer(); }

  [[nodiscard]] const KnownBits &getKnownBits(const SimplifyQuery &Q) const {
    if (!hasKnownBits())
      calculateKnownBits(Q);
    return Known;
  }

  [[nodiscard]] bool hasKnownBits() const { return Pointer.getInt(); }

  operator PointerType() const { return Pointer.getPointer(); }
  PointerType operator->() const { return Pointer.getPointer(); }
  ReferenceType operator*() const { return *Pointer.getPointer(); }
};

/// Specialization of WithCache<T*> that allows a WithCache<const T*> reference
/// to bind to it, allowing similar implicit conversion semantics as T* has when
/// passed to an API taking a const T*.
template <typename Arg>
class WithCache<Arg,
                std::enable_if_t<!std::is_const_v<std::remove_pointer_t<Arg>>>>
    : public WithCache<
          std::add_pointer_t<std::add_const_t<std::remove_pointer_t<Arg>>>> {
  using UnderlyingType = std::remove_pointer_t<Arg>;
  using Base = WithCache<const UnderlyingType *>;
  using PointerType = Arg;
  using ReferenceType = UnderlyingType &;

public:
  WithCache(PointerType Pointer) : Base(Pointer) {}
  WithCache(PointerType Pointer, const KnownBits &Known)
      : Base(Pointer, Known) {}

  [[nodiscard]] PointerType getValue() const {
    return const_cast<PointerType>(Base::getValue());
  }

  operator PointerType() const { return getValue(); }
  PointerType operator->() const { return getValue(); }
  ReferenceType operator*() const { return *getValue(); }
};

} // namespace llvm

#endif
