//===- ArmSMEOpInterfaces.h - Arm SME Dialect OpInterfaces ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSME_OPINTERFACES_H
#define MLIR_DIALECT_ARMSME_OPINTERFACES_H

#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace mlir::arm_sme {

namespace detail {
LogicalResult verifyArmSMETileOpInterface(Operation *);
}

static constexpr unsigned kInMemoryTileIdBase = 16;
#include "mlir/Dialect/ArmSME/IR/ArmSMEOpInterfaces.h.inc"
} // namespace mlir::arm_sme

#endif // MLIR_DIALECT_ARMSME_OPINTERFACES_H
