//===- OpenMPSIMDInlineBoost.cpp
//-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark function calls inside OpenMP SIMD regions with omp.simd_inline_boost
// so FIR-to-LLVM conversion can add an LLVM inline-threshold bonus, enabling
// more aggressive inlining for vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace omp {

#define GEN_PASS_DEF_OPENMPSIMDINLINEBOOSTPASS
#include "mlir/Dialect/OpenMP/Transforms/Passes.h.inc"

} // namespace omp
} // namespace mlir

using namespace mlir;
namespace {

class OpenMPSIMDInlineBoostPass
    : public omp::impl::OpenMPSIMDInlineBoostPassBase<
          OpenMPSIMDInlineBoostPass> {

  void runOnOperation() override {
    getOperation()->walk([](omp::SimdOp simdOp) {
      simdOp->walk([](CallOpInterface callOp) {
        Operation *op = callOp.getOperation();
        if (op->hasAttr("omp.simd_inline_boost"))
          return;
        op->setAttr("omp.simd_inline_boost", UnitAttr::get(op->getContext()));
      });
    });
  }
};

} // namespace
