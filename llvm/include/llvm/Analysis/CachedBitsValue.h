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

#include "llvm/IR/Value.h"
#include "llvm/Support/KnownBits.h"
#include <type_traits>

namespace llvm {

class DataLayout;
class AssumptionCache;
class Instruction;
class DominatorTree;
struct SimplifyQuery;

KnownBits computeKnownBits(const Value *V, const APInt &DemandedElts,
                           unsigned Depth, const SimplifyQuery &Q);

KnownBits computeKnownBits(const Value *V, unsigned Depth,
                           const SimplifyQuery &Q);

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

  ValuePointerType Pointer;
  mutable std::optional<KnownBits> Known;

  void calculateKnownBits(unsigned Depth, const SimplifyQuery &Q) const {
    Known = computeKnownBits(Pointer, Depth, Q);
  }

  void calculateKnownBits(const APInt &DemandedElts, unsigned Depth,
                          const SimplifyQuery &Q) const {
    Known = computeKnownBits(Pointer, DemandedElts, Depth, Q);
  }

public:
  ImplCachedBitsValue() = default;
  ImplCachedBitsValue(ValuePointerType Pointer)
      : Pointer(Pointer), Known(std::nullopt) {}
  ImplCachedBitsValue(ValuePointerType Pointer, const KnownBits &Known)
      : Pointer(Pointer), Known(Known) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  ImplCachedBitsValue(const T &Value)
      : Pointer(static_cast<ValuePointerType>(Value)), Known(std::nullopt) {}

  template <typename T, std::enable_if_t<ValuePointerConvertible<T>, int> = 0>
  ImplCachedBitsValue(const T &Value, const KnownBits &Known)
      : Pointer(static_cast<ValuePointerType>(Value)), Known(Known) {}

  [[nodiscard]] ValuePointerType getValue() { return Pointer; }
  [[nodiscard]] ValuePointerType getValue() const { return Pointer; }

  [[nodiscard]] const KnownBits &getKnownBits(unsigned Depth,
                                              const SimplifyQuery &Q) const {
    if (!hasKnownBits())
      calculateKnownBits(Depth, Q);
    return Known.value();
  }

  [[nodiscard]] KnownBits &getKnownBits(unsigned Depth,
                                        const SimplifyQuery &Q) {
    if (!hasKnownBits())
      calculateKnownBits(Depth, Q);
    return Known.value();
  }

  [[nodiscard]] const KnownBits &getKnownBits(const APInt &DemandedElts,
                                              unsigned Depth,
                                              const SimplifyQuery &Q) const {
    if (!hasKnownBits())
      calculateKnownBits(DemandedElts, Depth, Q);
    return Known.value();
  }

  [[nodiscard]] KnownBits &getKnownBits(const APInt &DemandedElts,
                                        unsigned Depth,
                                        const SimplifyQuery &Q) {
    if (!hasKnownBits())
      calculateKnownBits(DemandedElts, Depth, Q);
    return Known.value();
  }

  [[nodiscard]] bool hasKnownBits() const { return Known.has_value(); }

  operator ValuePointerType() { return Pointer; }
  ValuePointerType operator->() { return Pointer; }
  ValueReferenceType operator*() { return *Pointer; }

  operator ValuePointerType() const { return Pointer; }
  ValuePointerType operator->() const { return Pointer; }
  ValueReferenceType operator*() const { return *Pointer; }
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
      return CachedBitsConstValue(getValue(), Known.value());
    else
      return CachedBitsConstValue(getValue());
  }
};

} // namespace llvm

#endif
