//===- AffineInterfaceImpl.cpp - Affine Impl. of BufferizableOpInterface --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/ComprehensiveBufferize/AffineInterfaceImpl.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"

using namespace mlir::bufferization;

void mlir::linalg::comprehensive_bufferize::affine_ext::
    registerBufferizableOpInterfaceExternalModels(DialectRegistry &registry) {
  // AffineParallelOp bufferization not implemented yet. However, never hoist
  // memref allocations across AffineParallelOp boundaries.
  registry.addOpInterface<AffineParallelOp,
                          AllocationHoistingBarrierOnly<AffineParallelOp>>();
}
