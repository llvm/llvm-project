//===- TestLowerToArmSME.cpp - Test lowering to ArmSME as a sink pass -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass for testing the lowering to ArmSME as a
// generally usable sink pass.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/Vector/Transforms/Passes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TestLowerToArmSMEOptions
    : public PassPipelineOptions<TestLowerToArmSMEOptions> {
  PassOptions::Option<bool> fuseOuterProducts{
      *this, "fuse-outer-products",
      llvm::cl::desc("Fuse outer product operations via "
                     "'-arm-sme-outer-product-fusion' pass"),
      llvm::cl::init(true)};
};

void buildTestLowerToArmSME(OpPassManager &pm,
                            const TestLowerToArmSMEOptions &options) {
  // Lower 'vector.mask' operations.
  pm.addPass(vector::createLowerVectorMaskPass());

  // One shot bufferize. Convert ops with tensor semantics to ops with memref
  // semantics in a single pass.
  bufferization::OneShotBufferizationOptions bufferizationOptions;
  bufferizationOptions.bufferizeFunctionBoundaries = true;
  pm.addPass(bufferization::createOneShotBufferizePass(bufferizationOptions));

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());

  // Legalize vector operations so they can be converted to ArmSME.
  pm.addPass(arm_sme::createVectorLegalizationPass());

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // Passes that convert operations on vectors to ArmSME operations.

  // Convert Arith to ArmSME.
  pm.addPass(createArithToArmSMEConversionPass());
  // Convert Vector to ArmSME.
  pm.addPass(createConvertVectorToArmSMEPass());

  // Fuse outer products.
  if (options.fuseOuterProducts)
    pm.addPass(arm_sme::createOuterProductFusionPass());

  // Convert operations on high-level vectors to loops.

  // Convert ArmSME to SCF.
  pm.addPass(createConvertArmSMEToSCFPass());
  // Convert Vector to SCF.
  pm.addPass(createConvertVectorToSCFPass());

  // Allocate tiles for ArmSME operations.
  //
  // Later passes may create further ArmSME ops that implement the
  // ArmSMETileOpInterface, but tiles are allocated for root operations,
  // all of which should now exist.
  pm.addPass(arm_sme::createTileAllocationPass());

  // Configure PSTATE.SM and PSTATE.ZA.
  pm.addPass(arm_sme::createEnableArmStreamingPass(
      arm_sme::ArmStreamingMode::StreamingLocally, arm_sme::ArmZaMode::NewZA,
      /*onlyIfRequiredByOps=*/true));

  // Legalize operations on SVE vector types.
  pm.addPass(arm_sve::createLegalizeVectorStoragePass());

  // Convert ArmSME to LLVM.
  pm.addPass(createConvertArmSMEToLLVMPass());

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  // The remaining passes are taken from the -test-lower-to-llvm pipeline.

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
  ConvertVectorToLLVMPassOptions lowerVectorToLLVMOptions{};
  lowerVectorToLLVMOptions.armSVE = true;
  pm.addPass(createConvertVectorToLLVMPass(lowerVectorToLLVMOptions));
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
void registerTestLowerToArmSME() {
  PassPipelineRegistration<TestLowerToArmSMEOptions>(
      "test-lower-to-arm-sme",
      "An example pipeline to lower operations on vectors (arith, vector) to "
      "LLVM via ArmSME.",
      buildTestLowerToArmSME);
}
} // namespace test
} // namespace mlir
