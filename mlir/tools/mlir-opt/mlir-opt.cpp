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

// Defined in the test directory, no public header.
namespace mlir {
void registerConvertToTargetEnvPass();
void registerPassManagerTestPass();
void registerPrintOpAvailabilityPass();
void registerShapeFunctionTestPasses();
void registerSideEffectTestPasses();
void registerSliceAnalysisTestPass();
void registerSymbolTestPasses();
void registerTestAffineDataCopyPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestAllReduceLoweringPass();
void registerTestFunc();
void registerTestGpuMemoryPromotionPass();
void registerTestLoopPermutationPass();
void registerTestMatchers();
void registerTestPrintDefUsePass();
void registerTestPrintNestingPass();
void registerTestReducer();
void registerTestSpirvEntryPointABIPass();
void registerTestSpirvGLSLCanonicalizationPass();
void registerTestSpirvModuleCombinerPass();
void registerTestTraitsPass();
void registerTosaTestQuantUtilAPIPass();
void registerVectorizerTestPass();

namespace test {
void registerConvertCallOpPass();
void registerInliner();
void registerMemRefBoundCheck();
void registerPatternsTestPass();
void registerSimpleParametricTilingPass();
void registerTestAffineLoopParametricTilingPass();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestConvVectorization();
void registerTestConvertGPUKernelToCubinPass();
void registerTestConvertGPUKernelToHsacoPass();
void registerTestDecomposeCallGraphTypes();
void registerTestDialect(DialectRegistry &);
void registerTestDominancePass();
void registerTestDynamicPipelinePass();
void registerTestExpandTanhPass();
void registerTestGpuParallelLoopMappingPass();
void registerTestInterfaces();
void registerTestLinalgCodegenStrategy();
void registerTestLinalgFusionTransforms();
void registerTestLinalgGreedyFusion();
void registerTestLinalgHoisting();
void registerTestLinalgTileAndFuseSequencePass();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestNumberOfBlockExecutionsPass();
void registerTestNumberOfOperationExecutionsPass();
void registerTestOpaqueLoc();
void registerTestPDLByteCodePass();
void registerTestPreparationPassWithAllowedMemrefResults();
void registerTestRecursiveTypesPass();
void registerTestSCFUtilsPass();
void registerTestSparsification();
void registerTestVectorConversions();
} // namespace test
} // namespace mlir

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  registerConvertToTargetEnvPass();
  registerPassManagerTestPass();
  registerPrintOpAvailabilityPass();
  registerShapeFunctionTestPasses();
  registerSideEffectTestPasses();
  registerSliceAnalysisTestPass();
  registerSymbolTestPasses();
  registerTestAffineDataCopyPass();
  registerTestAffineLoopUnswitchingPass();
  registerTestAllReduceLoweringPass();
  registerTestFunc();
  registerTestGpuMemoryPromotionPass();
  registerTestLoopPermutationPass();
  registerTestMatchers();
  registerTestPrintDefUsePass();
  registerTestPrintNestingPass();
  registerTestReducer();
  registerTestSpirvEntryPointABIPass();
  registerTestSpirvGLSLCanonicalizationPass();
  registerTestSpirvModuleCombinerPass();
  registerTestTraitsPass();
  registerVectorizerTestPass();
  registerTosaTestQuantUtilAPIPass();

  test::registerConvertCallOpPass();
  test::registerInliner();
  test::registerMemRefBoundCheck();
  test::registerPatternsTestPass();
  test::registerSimpleParametricTilingPass();
  test::registerTestAffineLoopParametricTilingPass();
  test::registerTestCallGraphPass();
  test::registerTestConstantFold();
#if MLIR_CUDA_CONVERSIONS_ENABLED
  test::registerTestConvertGPUKernelToCubinPass();
#endif
#if MLIR_ROCM_CONVERSIONS_ENABLED
  test::registerTestConvertGPUKernelToHsacoPass();
#endif
  test::registerTestConvVectorization();
  test::registerTestDecomposeCallGraphTypes();
  test::registerTestDominancePass();
  test::registerTestDynamicPipelinePass();
  test::registerTestExpandTanhPass();
  test::registerTestGpuParallelLoopMappingPass();
  test::registerTestInterfaces();
  test::registerTestLinalgCodegenStrategy();
  test::registerTestLinalgFusionTransforms();
  test::registerTestLinalgGreedyFusion();
  test::registerTestLinalgHoisting();
  test::registerTestLinalgTileAndFuseSequencePass();
  test::registerTestLinalgTransforms();
  test::registerTestLivenessPass();
  test::registerTestLoopFusion();
  test::registerTestLoopMappingPass();
  test::registerTestLoopUnrollingPass();
  test::registerTestMemRefDependenceCheck();
  test::registerTestMemRefStrideCalculation();
  test::registerTestNumberOfBlockExecutionsPass();
  test::registerTestNumberOfOperationExecutionsPass();
  test::registerTestOpaqueLoc();
  test::registerTestPDLByteCodePass();
  test::registerTestRecursiveTypesPass();
  test::registerTestSCFUtilsPass();
  test::registerTestSparsification();
  test::registerTestVectorConversions();
}
#endif

int main(int argc, char **argv) {
  registerAllPasses();
#ifdef MLIR_INCLUDE_TESTS
  registerTestPasses();
#endif
  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  return failed(MlirOptMain(argc, argv, "MLIR modular optimizer driver\n",
                            registry,
                            /*preloadDialectsInContext=*/false));
}
