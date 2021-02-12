//===- Intrinsics.h - MLIR EDSC Intrinsics for Math ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_MATH_EDSC_INTRINSICS_H_
#define MLIR_DIALECT_MATH_EDSC_INTRINSICS_H_

#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/EDSC/Builders.h"

namespace mlir {
namespace edsc {
namespace intrinsics {

using math_rsqrt = ValueBuilder<math::RsqrtOp>;
using math_tanh = ValueBuilder<math::TanhOp>;

} // namespace intrinsics
} // namespace edsc
} // namespace mlir

#endif // MLIR_DIALECT_MATH_EDSC_INTRINSICS_H_
