//===- mlir-opt.cpp - MLIR Optimizer Driver -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Main entry function for mlir-opt for when built as standalone binary.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

namespace mlir {
// Defined in the test directory, no public header.
void registerConvertCallOpPass();
void registerConvertToTargetEnvPass();
void registerInliner();
void registerMemRefBoundCheck();
void registerPassManagerTestPass();
void registerPatternsTestPass();
void registerPrintOpAvailabilityPass();
void registerSideEffectTestPasses();
void registerSimpleParametricTilingPass();
void registerSliceAnalysisTestPass();
void registerSymbolTestPasses();
void registerTestAffineDataCopyPass();
void registerTestAffineLoopParametricTilingPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestAllReduceLoweringPass();
void registerTestBufferPlacementPreparationPass();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestConvVectorization();
void registerTestConvertGPUKernelToCubinPass();
void registerTestConvertGPUKernelToHsacoPass();
void registerTestDominancePass();
void registerTestDialect(DialectRegistry &);
void registerTestDynamicPipelinePass();
void registerTestExpandTanhPass();
void registerTestFunc();
void registerTestGpuMemoryPromotionPass();
void registerTestGpuParallelLoopMappingPass();
void registerTestInterfaces();
void registerTestLinalgHoisting();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopPermutationPass();
void registerTestLoopUnrollingPass();
void registerTestMatchers();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestOpaqueLoc();
void registerTestPreparationPassWithAllowedMemrefResults();
void registerTestPrintDefUsePass();
void registerTestPrintNestingPass();
void registerTestRecursiveTypesPass();
void registerTestReducer();
void registerTestSpirvEntryPointABIPass();
void registerTestSCFUtilsPass();
void registerTestVectorConversions();
void registerVectorizerTestPass();
} // namespace mlir

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  registerConvertCallOpPass();
  registerConvertToTargetEnvPass();
  registerInliner();
  registerMemRefBoundCheck();
  registerPassManagerTestPass();
  registerPatternsTestPass();
  registerPrintOpAvailabilityPass();
  registerSideEffectTestPasses();
  registerSimpleParametricTilingPass();
  registerSliceAnalysisTestPass();
  registerSymbolTestPasses();
  registerTestAffineDataCopyPass();
  registerTestAllReduceLoweringPass();
  registerTestAffineLoopUnswitchingPass();
  registerTestLoopPermutationPass();
  registerTestCallGraphPass();
  registerTestConvVectorization();
  registerTestConstantFold();
#if MLIR_CUDA_CONVERSIONS_ENABLED
  registerTestConvertGPUKernelToCubinPass();
#endif
#if MLIR_ROCM_CONVERSIONS_ENABLED
  registerTestConvertGPUKernelToHsacoPass();
#endif
  registerTestAffineLoopParametricTilingPass();
  registerTestBufferPlacementPreparationPass();
  registerTestDominancePass();
  registerTestDynamicPipelinePass();
  registerTestFunc();
  registerTestExpandTanhPass();
  registerTestGpuMemoryPromotionPass();
  registerTestInterfaces();
  registerTestLinalgHoisting();
  registerTestLinalgTransforms();
  registerTestLivenessPass();
  registerTestLoopFusion();
  registerTestLoopMappingPass();
  registerTestLoopUnrollingPass();
  registerTestMatchers();
  registerTestMemRefDependenceCheck();
  registerTestMemRefStrideCalculation();
  registerTestOpaqueLoc();
  registerTestPreparationPassWithAllowedMemrefResults();
  registerTestPrintDefUsePass();
  registerTestPrintNestingPass();
  registerTestRecursiveTypesPass();
  registerTestReducer();
  registerTestGpuParallelLoopMappingPass();
  registerTestSpirvEntryPointABIPass();
  registerTestSCFUtilsPass();
  registerTestVectorConversions();
  registerVectorizerTestPass();
}
#endif

int main(int argc, char **argv) {
  registerAllDialects();
  registerAllPasses();
#ifdef MLIR_INCLUDE_TESTS
  registerTestPasses();
#endif
  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  registerTestDialect(registry);
#endif
  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
