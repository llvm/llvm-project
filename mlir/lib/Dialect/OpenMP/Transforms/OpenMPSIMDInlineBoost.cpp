//===- OpenMPSIMDInlineBoost.cpp
//-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Mark calls inside OpenMP SIMD regions with `omp.simd_inline_boost` so that
// FIR-to-LLVM conversion can attach an LLVM inline-threshold bonus to calls to
// functions containing `omp.declare_simd`, making them more likely to be
// inlined for vectorization.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
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

static bool calleeHasDeclareSimd(CallOpInterface callOp,
                                 SymbolTable &symTable) {
  auto callableRef = callOp.getCallableForCallee();
  if (!callableRef)
    return false;
  auto symRef = dyn_cast<SymbolRefAttr>(callableRef);
  if (!symRef)
    return false;
  auto *callee = symTable.lookup(symRef.getRootReference());
  if (!callee)
    return false;
  bool found = false;
  callee->walk([&](omp::DeclareSimdOp) {
    found = true;
    return WalkResult::interrupt();
  });
  return found;
}

class OpenMPSIMDInlineBoostPass
    : public omp::impl::OpenMPSIMDInlineBoostPassBase<
          OpenMPSIMDInlineBoostPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symTable(module);

    module->walk([&](omp::SimdOp simdOp) {
      simdOp->walk([&](CallOpInterface callOp) {
        Operation *op = callOp.getOperation();
        if (op->hasAttr("omp.simd_inline_boost"))
          return;
        if (!calleeHasDeclareSimd(callOp, symTable))
          return;
        op->setAttr("omp.simd_inline_boost", UnitAttr::get(op->getContext()));
      });
    });
  }
};

} // namespace
