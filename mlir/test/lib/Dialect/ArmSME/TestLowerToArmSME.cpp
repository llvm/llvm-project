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

#include "mlir/Conversion/ArithToArmSME/ArithToArmSME.h"
#include "mlir/Conversion/ArmSMEToLLVM/ArmSMEToLLVM.h"
#include "mlir/Conversion/ArmSMEToSCF/ArmSMEToSCF.h"
#include "mlir/Conversion/VectorToArmSME/VectorToArmSME.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/ArmSME/Transforms/Passes.h"
#include "mlir/Dialect/ArmSVE/Transforms/Passes.h"
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

  // Convert Vector to SCF (with full unroll enabled).
  pm.addPass(createConvertVectorToSCFPass(
      VectorTransferToSCFOptions().enableFullUnroll()));

  // Allocate tiles for ArmSME operations.
  //
  // Later passes may create further ArmSME ops that implement the
  // ArmSMETileOpInterface, but tiles are allocated for root operations,
  // all of which should now exist.
  pm.addPass(arm_sme::createTileAllocationPass());

  // Enable streaming-mode and ZA.
  pm.addPass(arm_sme::createEnableArmStreamingPass(
      arm_sme::ArmStreamingMode::StreamingLocally, arm_sme::ArmZaMode::NewZA,
      /*onlyIfRequiredByOps=*/true));

  // Convert ArmSME to LLVM.
  pm.addPass(createConvertArmSMEToLLVMPass());

  // Sprinkle some cleanups.
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
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
