//===- SlowDynamicAPInt.h - SlowDynamicAPInt Class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a simple class to represent arbitrary precision signed integers.
// Unlike APInt, one does not have to specify a fixed maximum size, and the
// integer can take on any arbitrary values.
//
// This class is to be used as a fallback slow path for the DynamicAPInt class,
// and is not intended to be used directly.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ADT_SLOWDYNAMICAPINT_H
#define LLVM_ADT_SLOWDYNAMICAPINT_H

#include "llvm/ADT/APInt.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class DynamicAPInt;
class raw_ostream;
} // namespace llvm

namespace llvm::detail {
/// A simple class providing dynamic arbitrary-precision arithmetic. Internally,
/// it stores an APInt, whose width is doubled whenever an overflow occurs at a
/// certain width. The default constructor sets the initial width to 64.
/// SlowDynamicAPInt is primarily intended to be used as a slow fallback path
/// for the upcoming DynamicAPInt class.
class SlowDynamicAPInt {
  APInt Val;

public:
  LLVM_ABI explicit SlowDynamicAPInt(int64_t Val);
  LLVM_ABI SlowDynamicAPInt();
  LLVM_ABI explicit SlowDynamicAPInt(const APInt &Val);
  LLVM_ABI SlowDynamicAPInt &operator=(int64_t Val);
  LLVM_ABI explicit operator int64_t() const;
  LLVM_ABI SlowDynamicAPInt operator-() const;
  LLVM_ABI bool operator==(const SlowDynamicAPInt &O) const;
  LLVM_ABI bool operator!=(const SlowDynamicAPInt &O) const;
  LLVM_ABI bool operator>(const SlowDynamicAPInt &O) const;
  LLVM_ABI bool operator<(const SlowDynamicAPInt &O) const;
  LLVM_ABI bool operator<=(const SlowDynamicAPInt &O) const;
  LLVM_ABI bool operator>=(const SlowDynamicAPInt &O) const;
  LLVM_ABI SlowDynamicAPInt operator+(const SlowDynamicAPInt &O) const;
  LLVM_ABI SlowDynamicAPInt operator-(const SlowDynamicAPInt &O) const;
  LLVM_ABI SlowDynamicAPInt operator*(const SlowDynamicAPInt &O) const;
  LLVM_ABI SlowDynamicAPInt operator/(const SlowDynamicAPInt &O) const;
  LLVM_ABI SlowDynamicAPInt operator%(const SlowDynamicAPInt &O) const;
  LLVM_ABI SlowDynamicAPInt &operator+=(const SlowDynamicAPInt &O);
  LLVM_ABI SlowDynamicAPInt &operator-=(const SlowDynamicAPInt &O);
  LLVM_ABI SlowDynamicAPInt &operator*=(const SlowDynamicAPInt &O);
  LLVM_ABI SlowDynamicAPInt &operator/=(const SlowDynamicAPInt &O);
  LLVM_ABI SlowDynamicAPInt &operator%=(const SlowDynamicAPInt &O);

  LLVM_ABI SlowDynamicAPInt &operator++();
  LLVM_ABI SlowDynamicAPInt &operator--();

  LLVM_ABI friend SlowDynamicAPInt abs(const SlowDynamicAPInt &X);
  LLVM_ABI friend SlowDynamicAPInt ceilDiv(const SlowDynamicAPInt &LHS,
                                           const SlowDynamicAPInt &RHS);
  LLVM_ABI friend SlowDynamicAPInt floorDiv(const SlowDynamicAPInt &LHS,
                                            const SlowDynamicAPInt &RHS);
  /// The operands must be non-negative for gcd.
  LLVM_ABI friend SlowDynamicAPInt gcd(const SlowDynamicAPInt &A,
                                       const SlowDynamicAPInt &B);

  /// Overload to compute a hash_code for a SlowDynamicAPInt value.
  LLVM_ABI friend hash_code hash_value(const SlowDynamicAPInt &X); // NOLINT

  // Make DynamicAPInt a friend so it can access Val directly.
  friend DynamicAPInt;

  unsigned getBitWidth() const { return Val.getBitWidth(); }

  LLVM_ABI void print(raw_ostream &OS) const;

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
  LLVM_DUMP_METHOD void dump() const;
#endif
};

inline raw_ostream &operator<<(raw_ostream &OS, const SlowDynamicAPInt &X) {
  X.print(OS);
  return OS;
}

/// Returns the remainder of dividing LHS by RHS.
///
/// The RHS is always expected to be positive, and the result
/// is always non-negative.
LLVM_ABI SlowDynamicAPInt mod(const SlowDynamicAPInt &LHS,
                              const SlowDynamicAPInt &RHS);

/// Returns the least common multiple of A and B.
LLVM_ABI SlowDynamicAPInt lcm(const SlowDynamicAPInt &A,
                              const SlowDynamicAPInt &B);

/// Redeclarations of friend declarations above to
/// make it discoverable by lookups.
LLVM_ABI SlowDynamicAPInt abs(const SlowDynamicAPInt &X);
LLVM_ABI SlowDynamicAPInt ceilDiv(const SlowDynamicAPInt &LHS,
                                  const SlowDynamicAPInt &RHS);
LLVM_ABI SlowDynamicAPInt floorDiv(const SlowDynamicAPInt &LHS,
                                   const SlowDynamicAPInt &RHS);
LLVM_ABI SlowDynamicAPInt gcd(const SlowDynamicAPInt &A,
                              const SlowDynamicAPInt &B);
LLVM_ABI hash_code hash_value(const SlowDynamicAPInt &X); // NOLINT

/// ---------------------------------------------------------------------------
/// Convenience operator overloads for int64_t.
/// ---------------------------------------------------------------------------
LLVM_ABI SlowDynamicAPInt &operator+=(SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt &operator-=(SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt &operator*=(SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt &operator/=(SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt &operator%=(SlowDynamicAPInt &A, int64_t B);

LLVM_ABI bool operator==(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI bool operator!=(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI bool operator>(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI bool operator<(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI bool operator<=(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI bool operator>=(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt operator+(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt operator-(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt operator*(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt operator/(const SlowDynamicAPInt &A, int64_t B);
LLVM_ABI SlowDynamicAPInt operator%(const SlowDynamicAPInt &A, int64_t B);

LLVM_ABI bool operator==(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI bool operator!=(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI bool operator>(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI bool operator<(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI bool operator<=(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI bool operator>=(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI SlowDynamicAPInt operator+(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI SlowDynamicAPInt operator-(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI SlowDynamicAPInt operator*(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI SlowDynamicAPInt operator/(int64_t A, const SlowDynamicAPInt &B);
LLVM_ABI SlowDynamicAPInt operator%(int64_t A, const SlowDynamicAPInt &B);
} // namespace llvm::detail

#endif // LLVM_ADT_SLOWDYNAMICAPINT_H
