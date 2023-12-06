//===- MathExtras.h - Math functions relevant to MLIR -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains math functions relevant to MLIR.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_SUPPORT_MATHEXTRAS_H_
#define MLIR_SUPPORT_MATHEXTRAS_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include <type_traits>

namespace mlir {

// ceilDiv for unsigned integral.
template <typename T, std::enable_if_t<std::is_integral_v<T> &&
                                       !std::is_unsigned_v<T>> = true>
T ceilDiv(T lhs, T rhs) {
  assert(rhs != static_cast<T>(0));
  T q = lhs / rhs;
  T r = lhs % rhs;
  return r == static_cast<T>(0) ? q : q + static_cast<T>(1);
}

/// Returns the result of MLIR's ceildiv operation on constants. The RHS is
/// expected to be non-zero.
inline int64_t ceilDiv(int64_t lhs, int64_t rhs) {
  assert(rhs != 0);
  // C/C++'s integer division rounds towards 0.
  int64_t x = (rhs > 0) ? -1 : 1;
  return ((lhs != 0) && (lhs > 0) == (rhs > 0)) ? ((lhs + x) / rhs) + 1
                                                : -(-lhs / rhs);
}

/// Returns the result of MLIR's floordiv operation on constants. The RHS is
/// expected to be non-zero.
inline int64_t floorDiv(int64_t lhs, int64_t rhs) {
  assert(rhs != 0);
  // C/C++'s integer division rounds towards 0.
  int64_t x = (rhs < 0) ? 1 : -1;
  return ((lhs != 0) && ((lhs < 0) != (rhs < 0))) ? -((-lhs + x) / rhs) - 1
                                                  : lhs / rhs;
}

/// Returns MLIR's mod operation on constants. MLIR's mod operation yields the
/// remainder of the Euclidean division of 'lhs' by 'rhs', and is therefore not
/// C's % operator.  The RHS is always expected to be positive, and the result
/// is always non-negative.
inline int64_t mod(int64_t lhs, int64_t rhs) {
  assert(rhs >= 1);
  return lhs % rhs < 0 ? lhs % rhs + rhs : lhs % rhs;
}
} // namespace mlir

#endif // MLIR_SUPPORT_MATHEXTRAS_H_
