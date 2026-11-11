//===- CompositePass.h - Composite pass utilities ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_COMPOSITEPASS_H
#define MLIR_TRANSFORMS_COMPOSITEPASS_H

namespace mlir {

/// Action to take when `CompositeFixedPointPass` fails to converge within
/// its configured maximum number of iterations.
enum class ConvergenceFailureAction {
  /// Emit a warning.
  Warn,
  /// Emit an error and fail the pass.
  Error,
  /// Do nothing.
  Silent,
};

} // namespace mlir

#endif // MLIR_TRANSFORMS_COMPOSITEPASS_H
