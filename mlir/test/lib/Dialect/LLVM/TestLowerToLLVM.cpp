//===- TestLowerToLLVM.cpp - Test lowering to LLVM as a sink pass ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to LLVM as a generally
// usable sink pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TestLowerToLLVMOptions
    : public PassPipelineOptions<TestLowerToLLVMOptions> {
  PassOptions::Option<bool> reassociateFPReductions{
      *this, "reassociate-fp-reductions",
      llvm::cl::desc("Allow reassociation og FP reductions"),
      llvm::cl::init(false)};
};

void buildTestLowerToLLVM(OpPassManager &pm,
                          const TestLowerToLLVMOptions &options) {

  // TODO: it is feasible to scope lowering at arbitrary level and introduce
  // unrealized casts, but there needs to be the final module-wise cleanup in
  // the end. Keep module-level for now.

  // Blanket-convert any remaining high-level vector ops to loops if any remain.
  pm.addNestedPass<func::FuncOp>(createConvertVectorToSCFPass());
  // Blanket-convert any remaining linalg ops to loops if any remain.
  pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());
  // Blanket-convert any remaining affine ops if any remain.
  pm.addPass(createLowerAffinePass());
  // Convert SCF to CF (always needed).
  pm.addPass(createConvertSCFToCFPass());
  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  // Convert vector to LLVM (always needed).
  pm.addPass(createConvertVectorToLLVMPass(
      // TODO: add more options on a per-need basis.
      ConvertVectorToLLVMPassOptions{options.reassociateFPReductions}));
  // Convert Math to LLVM (always needed).
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  // Expand complicated MemRef operations before lowering them.
  pm.addPass(memref::createExpandStridedMetadataPass());
  // The expansion may create affine expressions. Get rid of them.
  pm.addPass(createLowerAffinePass());
  // Convert MemRef to LLVM (always needed).
  pm.addPass(createFinalizeMemRefToLLVMConversionPass());
  // Convert Func to LLVM (always needed).
  pm.addPass(createConvertFuncToLLVMPass());
  // Convert Index to LLVM (always needed).
  pm.addPass(createConvertIndexToLLVMPass());
  // Convert remaining unrealized_casts (always needed).
  pm.addPass(createReconcileUnrealizedCastsPass());
}
} // namespace

namespace mlir {
namespace test {
void registerTestLowerToLLVM() {
  PassPipelineRegistration<TestLowerToLLVMOptions>(
      "test-lower-to-llvm",
      "An example of pipeline to lower the main dialects (arith, linalg, "
      "memref, scf, vector) down to LLVM.",
      buildTestLowerToLLVM);
}
} // namespace test
} // namespace mlir
