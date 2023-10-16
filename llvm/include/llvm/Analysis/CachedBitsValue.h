//===- llvm/Analysis/CachedBitsValue.h - Value with KnownBits - -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Store a pointer to an llvm::Value along with the KnownBits information for it
// that is computed lazily (if required).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_CACHEDBITSVALUE_H
#define LLVM_ANALYSIS_CACHEDBITSVALUE_H

#include "llvm/ADT/PointerIntPair.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/KnownBits.h"
#include <type_traits>


namespace llvm {
  struct SimplifyQuery;
}

llvm::KnownBits computeKnownBits(const llvm::Value *V, unsigned Depth,
                                 const llvm::SimplifyQuery &Q);

namespace llvm {
namespace detail {
/// Represents a pointer to an llvm::Value with known bits information
template <bool ConstPointer = true> class ImplCachedBitsValue {
protected:
  using ValuePointerType =
      std::conditional_t<ConstPointer, const Value *, Value *>;
  using ValueReferenceType =
      std::conditional_t<ConstPointer, const Value &, Value &>;

  template <typename T>
  constexpr static bool ValuePointerConvertible =
      std::is_convertible_v<T, ValuePointerType>;

  // Store the presence of the KnownBits information in one of the bits of
  // Pointer.
  // true  -> present
  // false -> absent
  mutable PointerIntPair<ValuePointerType, 1, bool> Pointer;
  mutable KnownBits Known;

  void calculateKnownBits(const SimplifyQuery &Q) const {
    Known = computeKnownBits(Pointer.getPointer(), 0, Q);
    Pointer.setInt(true);
  }

public:
  ImplCachedBitsValue() = default;
  ImplCachedBitsValue(ValuePointerType Pointer) : Pointer(Pointer, false) {}
  ImplCachedBitsValue(ValuePointerType Pointer, const KnownBits &Known)
      : Pointer(Pointer, true), Known(Known) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  ImplCachedBitsValue(const T &Value)
      : Pointer(static_cast<ValuePointerType>(Value), false) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  ImplCachedBitsValue(const T &Value, const KnownBits &Known)
      : Pointer(static_cast<ValuePointerType>(Value), true), Known(Known) {}

  [[nodiscard]] ValuePointerType getValue() { return Pointer.getPointer(); }
  [[nodiscard]] ValuePointerType getValue() const {
    return Pointer.getPointer();
  }

  [[nodiscard]] const KnownBits &getKnownBits(const SimplifyQuery &Q) const {
    if (!hasKnownBits())
      calculateKnownBits(Q);
    return Known;
  }

  [[nodiscard]] KnownBits &getKnownBits(const SimplifyQuery &Q) {
    if (!hasKnownBits())
      calculateKnownBits(Q);
    return Known;
  }

  [[nodiscard]] bool hasKnownBits() const { return Pointer.getInt(); }

  operator ValuePointerType() { return Pointer.getPointer(); }
  ValuePointerType operator->() { return Pointer.getPointer(); }
  ValueReferenceType operator*() { return *Pointer.getPointer(); }

  operator ValuePointerType() const { return Pointer.getPointer(); }
  ValuePointerType operator->() const { return Pointer.getPointer(); }
  ValueReferenceType operator*() const { return *Pointer.getPointer(); }
};
} // namespace detail

class CachedBitsConstValue : public detail::ImplCachedBitsValue<true> {
public:
  CachedBitsConstValue() = default;
  CachedBitsConstValue(ValuePointerType Pointer)
      : ImplCachedBitsValue(Pointer) {}
  CachedBitsConstValue(Value *Pointer) : ImplCachedBitsValue(Pointer) {}
  CachedBitsConstValue(ValuePointerType Pointer, const KnownBits &Known)
      : ImplCachedBitsValue(Pointer, Known) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  CachedBitsConstValue(const T &Value) : ImplCachedBitsValue(Value) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  CachedBitsConstValue(const T &Value, const KnownBits &Known)
      : ImplCachedBitsValue(Value, Known) {}
};

class CachedBitsNonConstValue : public detail::ImplCachedBitsValue<false> {
public:
  CachedBitsNonConstValue() = default;
  CachedBitsNonConstValue(ValuePointerType Pointer)
      : ImplCachedBitsValue(Pointer) {}
  CachedBitsNonConstValue(ValuePointerType Pointer, const KnownBits &Known)
      : ImplCachedBitsValue(Pointer, Known) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  CachedBitsNonConstValue(const T &Value) : ImplCachedBitsValue(Value) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  CachedBitsNonConstValue(const T &Value, const KnownBits &Known)
      : ImplCachedBitsValue(Value, Known) {}

  [[nodiscard]] CachedBitsConstValue toConst() const {
    if (hasKnownBits())
      return CachedBitsConstValue(getValue(), Known);
    else
      return CachedBitsConstValue(getValue());
  }
};

} // namespace llvm

#endif
