//===- TosaProfileCompliance.h - Tosa Profile-based Compliance Validation -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H
#define MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H

#include "mlir/Dialect/Tosa/IR/TargetEnv.h"

namespace mlir {
namespace tosa {
//===----------------------------------------------------------------------===//
// Tosa Profile And Extension Compliance Checker
//===----------------------------------------------------------------------===//

class TosaProfileCompliance {
public:
  explicit TosaProfileCompliance();

  // Verify if the operation is allowed to be executed in the given target
  // environment.
  LogicalResult checkProfile(Operation *op, const tosa::TargetEnv &targetEnv);
  LogicalResult checkExtension(Operation *op, const tosa::TargetEnv &targetEnv);
  LogicalResult checkInvalid(Operation *op);
};
} // namespace tosa
} // namespace mlir

#endif // MLIR_DIALECT_TOSA_TRANSFORMS_TOSAPROFILECOMPILANCE_H
