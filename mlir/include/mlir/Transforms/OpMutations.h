//===- OpMutations.h - Location Snapshot Utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file several utility methods for snapshotting the current IR to
// produce new debug locations.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TRANSFORMS_OPMUTATIONS_H
#define MLIR_TRANSFORMS_OPMUTATIONS_H

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"

#include <memory>

namespace mlir {
class IRMapping;
class Operation;
class Pass;

#define GEN_PASS_DECL_PRINTOPMUTATIONS
#include "mlir/Transforms/Passes.h.inc"

void getOpMutations(Operation *op_before, Operation *op_after,
                    const IRMapping &ir_map);

/// Overload utilizing pass options for initialization.
std::unique_ptr<Pass> createPrintOpMutationsPass();

} // namespace mlir

#endif // MLIR_TRANSFORMS_OPMUTATIONS_H
