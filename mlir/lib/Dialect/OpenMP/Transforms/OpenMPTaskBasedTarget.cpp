//===- OpenMPTaskBasedTarget.cpp - Implementation of OpenMPTaskBasedTargetPass
//---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements scf.parallel to scf.for + async.execute conversion pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"

namespace mlir {
#define GEN_PASS_DEF_OPENMPTASKBASEDTARGET
#include "mlir/Dialect/OpenMP/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::omp;

#define DEBUG_TYPE "openmp-task-based-target"

namespace {

struct OpenMPTaskBasedTargetPass
    : public impl::OpenMPTaskBasedTargetBase<OpenMPTaskBasedTargetPass> {

  void runOnOperation() override;
};

} // namespace

void OpenMPTaskBasedTargetPass::runOnOperation() {
  Operation *op = getOperation();

  op->dump();
}
std::unique_ptr<Pass> mlir::createOpenMPTaskBasedTargetPass() {
  return std::make_unique<OpenMPTaskBasedTargetPass>();
}
