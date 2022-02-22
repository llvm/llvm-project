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
void registerPrintSpirvAvailabilityPass();
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
void registerTestOperationEqualPass();
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
void registerTestAliasAnalysisPass();
void registerTestBuiltinAttributeInterfaces();
void registerTestCallGraphPass();
void registerTestConstantFold();
void registerTestGpuSerializeToCubinPass();
void registerTestGpuSerializeToHsacoPass();
void registerTestDataLayoutQuery();
void registerTestDecomposeCallGraphTypes();
void registerTestDiagnosticsPass();
void registerTestDominancePass();
void registerTestDynamicPipelinePass();
void registerTestExpandTanhPass();
void registerTestComposeSubView();
void registerTestGpuParallelLoopMappingPass();
void registerTestIRVisitorsPass();
void registerTestGenericIRVisitorsPass();
void registerTestGenericIRVisitorsInterruptPass();
void registerTestInterfaces();
void registerTestLinalgCodegenStrategy();
void registerTestLinalgDistribution();
void registerTestLinalgElementwiseFusion();
void registerTestLinalgFusionTransforms();
void registerTestLinalgTensorFusionTransforms();
void registerTestLinalgTiledLoopFusionTransforms();
void registerTestLinalgGreedyFusion();
void registerTestLinalgHoisting();
void registerTestLinalgTileAndFuseSequencePass();
void registerTestLinalgTransforms();
void registerTestLivenessPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestMatchReductionPass();
void registerTestMathAlgebraicSimplificationPass();
void registerTestMathPolynomialApproximationPass();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestOpaqueLoc();
void registerTestPadFusion();
void registerTestPDLByteCodePass();
void registerTestPreparationPassWithAllowedMemrefResults();
void registerTestRecursiveTypesPass();
void registerTestSCFUtilsPass();
void registerTestSliceAnalysisPass();
void registerTestTensorTransforms();
void registerTestVectorLowerings();
} // namespace test
} // namespace mlir

namespace test {
void registerTestDialect(DialectRegistry &);
} // namespace test

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  registerConvertToTargetEnvPass();
  registerPassManagerTestPass();
  registerPrintSpirvAvailabilityPass();
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
  registerTestOperationEqualPass();
  registerTestPrintDefUsePass();
  registerTestPrintNestingPass();
  registerTestReducer();
  registerTestSpirvEntryPointABIPass();
  registerTestSpirvGLSLCanonicalizationPass();
  registerTestSpirvModuleCombinerPass();
  registerTestTraitsPass();
  registerVectorizerTestPass();
  registerTosaTestQuantUtilAPIPass();

  mlir::test::registerConvertCallOpPass();
  mlir::test::registerInliner();
  mlir::test::registerMemRefBoundCheck();
  mlir::test::registerPatternsTestPass();
  mlir::test::registerSimpleParametricTilingPass();
  mlir::test::registerTestAffineLoopParametricTilingPass();
  mlir::test::registerTestAliasAnalysisPass();
  mlir::test::registerTestBuiltinAttributeInterfaces();
  mlir::test::registerTestCallGraphPass();
  mlir::test::registerTestConstantFold();
  mlir::test::registerTestDiagnosticsPass();
#if MLIR_CUDA_CONVERSIONS_ENABLED
  mlir::test::registerTestGpuSerializeToCubinPass();
#endif
#if MLIR_ROCM_CONVERSIONS_ENABLED
  mlir::test::registerTestGpuSerializeToHsacoPass();
#endif
  mlir::test::registerTestDecomposeCallGraphTypes();
  mlir::test::registerTestDataLayoutQuery();
  mlir::test::registerTestDominancePass();
  mlir::test::registerTestDynamicPipelinePass();
  mlir::test::registerTestExpandTanhPass();
  mlir::test::registerTestComposeSubView();
  mlir::test::registerTestGpuParallelLoopMappingPass();
  mlir::test::registerTestIRVisitorsPass();
  mlir::test::registerTestGenericIRVisitorsPass();
  mlir::test::registerTestInterfaces();
  mlir::test::registerTestLinalgCodegenStrategy();
  mlir::test::registerTestLinalgDistribution();
  mlir::test::registerTestLinalgElementwiseFusion();
  mlir::test::registerTestLinalgFusionTransforms();
  mlir::test::registerTestLinalgTensorFusionTransforms();
  mlir::test::registerTestLinalgTiledLoopFusionTransforms();
  mlir::test::registerTestLinalgGreedyFusion();
  mlir::test::registerTestLinalgHoisting();
  mlir::test::registerTestLinalgTileAndFuseSequencePass();
  mlir::test::registerTestLinalgTransforms();
  mlir::test::registerTestLivenessPass();
  mlir::test::registerTestLoopFusion();
  mlir::test::registerTestLoopMappingPass();
  mlir::test::registerTestLoopUnrollingPass();
  mlir::test::registerTestMatchReductionPass();
  mlir::test::registerTestMathAlgebraicSimplificationPass();
  mlir::test::registerTestMathPolynomialApproximationPass();
  mlir::test::registerTestMemRefDependenceCheck();
  mlir::test::registerTestMemRefStrideCalculation();
  mlir::test::registerTestOpaqueLoc();
  mlir::test::registerTestPadFusion();
  mlir::test::registerTestPDLByteCodePass();
  mlir::test::registerTestRecursiveTypesPass();
  mlir::test::registerTestSCFUtilsPass();
  mlir::test::registerTestSliceAnalysisPass();
  mlir::test::registerTestTensorTransforms();
  mlir::test::registerTestVectorLowerings();
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
  ::test::registerTestDialect(registry);
#endif
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MLIR modular optimizer driver\n", registry,
                        /*preloadDialectsInContext=*/false));
}
