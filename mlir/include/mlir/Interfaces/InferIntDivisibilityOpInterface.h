//===- InferIntDivisibilityOpInterface.h - Integer Divisibility -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains definitions of the integer divisibility inference
// interface defined in `InferIntDivisibilityOpInterface.td`.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_INFERINTDIVISIBILITYOPINTERFACE_H
#define MLIR_INTERFACES_INFERINTDIVISIBILITYOPINTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include <numeric>
#include <optional>

namespace mlir {

/// Statically known divisibility information for an integer SSA value.
/// Tracks separate divisors for the unsigned and signed interpretations of
/// the value so that subsequent analyses can use whichever is more precise.
class ConstantIntDivisibility {
public:
  ConstantIntDivisibility() = default;
  ConstantIntDivisibility(uint64_t udiv, uint64_t sdiv)
      : udivVal(udiv), sdivVal(sdiv) {}

  bool operator==(const ConstantIntDivisibility &other) const {
    return udivVal == other.udivVal && sdivVal == other.sdivVal;
  }

  uint64_t udiv() const { return this->udivVal; }
  uint64_t sdiv() const { return this->sdivVal; }

  // Returns the union (computed separately for signed and unsigned bounds)
  // for this divisibility and `other`.
  ConstantIntDivisibility getUnion(const ConstantIntDivisibility &other) const {
    return ConstantIntDivisibility(
        /*udiv=*/std::gcd(udiv(), other.udiv()),
        /*sdiv=*/std::gcd(sdiv(), other.sdiv()));
  }

private:
  uint64_t udivVal;
  uint64_t sdivVal;

  friend raw_ostream &operator<<(raw_ostream &os,
                                 const ConstantIntDivisibility &div);
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const ConstantIntDivisibility &div) {
  os << "ConstantIntDivisibility(udiv = " << div.udivVal
     << ", sdiv = " << div.sdivVal << ")";
  return os;
}

/// This lattice value represents the integer divisibility of an SSA value.
class IntegerDivisibility {
public:
  IntegerDivisibility(ConstantIntDivisibility value)
      : value(std::move(value)) {}
  explicit IntegerDivisibility(
      std::optional<ConstantIntDivisibility> value = std::nullopt)
      : value(std::move(value)) {}
  // Gets the minimum divisibility of 1 that is used to indicate that the value
  // cannot be analyzed further.
  static IntegerDivisibility getMinDivisibility() {
    return IntegerDivisibility(ConstantIntDivisibility(1, 1));
  }

  bool isUninitialized() const { return !value.has_value(); }
  const ConstantIntDivisibility &getValue() const {
    assert(!isUninitialized());
    return *value;
  }

  bool operator==(const IntegerDivisibility &rhs) const {
    return value == rhs.value;
  }

  static IntegerDivisibility join(const IntegerDivisibility &lhs,
                                  const IntegerDivisibility &rhs) {
    if (lhs.isUninitialized()) {
      return rhs;
    }
    if (rhs.isUninitialized()) {
      return lhs;
    }
    return IntegerDivisibility(lhs.getValue().getUnion(rhs.getValue()));
  }

  void print(raw_ostream &os) const { os << value; }

private:
  std::optional<ConstantIntDivisibility> value;
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const IntegerDivisibility &div) {
  div.print(os);
  return os;
}

/// The type of the `setResultDivs` callback provided to ops implementing
/// InferIntDivisibilityOpInterface. It should be called once for each integer
/// result value and be passed the ConstantIntDivisibility corresponding to
/// that value.
using SetIntDivisibilityFn =
    llvm::function_ref<void(Value, const ConstantIntDivisibility &)>;

} // end namespace mlir

#include "mlir/Interfaces/InferIntDivisibilityOpInterface.h.inc"

#endif // MLIR_INTERFACES_INFERINTDIVISIBILITYOPINTERFACE_H
