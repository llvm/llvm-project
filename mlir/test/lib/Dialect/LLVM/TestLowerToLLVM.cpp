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
#include "mlir/Conversion/LinalgToLLVM/LinalgToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TestLowerToLLVM
    : public PassWrapper<TestLowerToLLVM, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestLowerToLLVM)

  TestLowerToLLVM() = default;
  TestLowerToLLVM(const TestLowerToLLVM &pass) : PassWrapper(pass) {}

  StringRef getArgument() const final { return "test-lower-to-llvm"; }
  StringRef getDescription() const final {
    return "Test lowering to LLVM as a generally usable sink pass";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  Option<bool> reassociateFPReductions{
      *this, "reassociate-fp-reductions",
      llvm::cl::desc("Allow reassociation og FP reductions"),
      llvm::cl::init(false)};

  void runOnOperation() final;
};
} // namespace

void TestLowerToLLVM::runOnOperation() {
  MLIRContext *context = &this->getContext();
  RewritePatternSet patterns(context);

  // TODO: it is feasible to scope lowering at arbitrary level and introduce
  // unrealized casts, but there needs to be the final module-wise cleanup in
  // the end. Keep module-level for now.
  PassManager pm(&getContext());

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
  // Blanket-convert any remaining linalg ops to LLVM if any remain.
  pm.addPass(createConvertLinalgToLLVMPass());
  // Convert vector to LLVM (always needed).
  pm.addPass(createConvertVectorToLLVMPass(
      // TODO: add more options on a per-need basis.
      LowerVectorToLLVMOptions().enableReassociateFPReductions(
          reassociateFPReductions)));
  // Convert Math to LLVM (always needed).
  pm.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  // Convert MemRef to LLVM (always needed).
  pm.addPass(createMemRefToLLVMConversionPass());
  // Convert Func to LLVM (always needed).
  pm.addPass(createConvertFuncToLLVMPass());
  // Convert Index to LLVM (always needed).
  pm.addPass(createConvertIndexToLLVMPass());
  // Convert remaining unrealized_casts (always needed).
  pm.addPass(createReconcileUnrealizedCastsPass());
  if (failed(pm.run(getOperation()))) {
    getOperation()->dump();
    return signalPassFailure();
  }
}

namespace mlir {
namespace test {
void registerTestLowerToLLVM() { PassRegistration<TestLowerToLLVM>(); }
} // namespace test
} // namespace mlir
