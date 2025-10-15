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

#include "mlir/Config/mlir-config.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace llvm;
using namespace mlir;

// Defined in the test directory, no public header.
namespace mlir {
void registerCloneTestPasses();
void registerConvertToTargetEnvPass();
void registerLazyLoadingTestPasses();
void registerLoopLikeInterfaceTestPasses();
void registerPassManagerTestPass();
void registerPrintSpirvAvailabilityPass();
void registerRegionTestPasses();
void registerPrintTosaAvailabilityPass();
void registerShapeFunctionTestPasses();
void registerSideEffectTestPasses();
void registerSliceAnalysisTestPass();
void registerSymbolTestPasses();
void registerTestAffineAccessAnalysisPass();
void registerTestAffineDataCopyPass();
void registerTestAffineLoopUnswitchingPass();
void registerTestAffineReifyValueBoundsPass();
void registerTestAffineWalk();
void registerTestBytecodeRoundtripPasses();
void registerTestDecomposeAffineOpPass();
void registerTestFunc();
void registerTestGpuLoweringPasses();
void registerTestGpuMemoryPromotionPass();
void registerTestLoopPermutationPass();
void registerTestMatchers();
void registerTestOperationEqualPass();
void registerTestPreserveUseListOrders();
void registerTestPrintDefUsePass();
void registerTestPrintInvalidPass();
void registerTestPrintNestingPass();
void registerTestReducer();
void registerTestSpirvEntryPointABIPass();
void registerTestSpirvModuleCombinerPass();
void registerTestTraitsPass();
void registerTosaTestQuantUtilAPIPass();
void registerVectorizerTestPass();

namespace test {
void registerCommutativityUtils();
void registerConvertCallOpPass();
void registerConvertFuncOpPass();
void registerInliner();
void registerInlinerCallback();
void registerMemRefBoundCheck();
void registerPatternsTestPass();
void registerSimpleParametricTilingPass();
void registerTestAffineLoopParametricTilingPass();
void registerTestAliasAnalysisPass();
void registerTestArithEmulateWideIntPass();
void registerTestBuiltinAttributeInterfaces();
void registerTestBuiltinDistinctAttributes();
void registerTestCallGraphPass();
void registerTestCfAssertPass();
void registerTestCFGLoopInfoPass();
void registerTestComposeSubView();
void registerTestCompositePass();
void registerTestControlFlowSink();
void registerTestConvertToSPIRVPass();
void registerTestDataLayoutPropagation();
void registerTestDataLayoutQuery();
void registerTestDeadCodeAnalysisPass();
void registerTestDecomposeCallGraphTypes();
void registerTestDiagnosticsPass();
void registerTestDiagnosticsMetadataPass();
void registerTestDominancePass();
void registerTestDynamicPipelinePass();
void registerTestRemarkPass();
void registerTestEmulateNarrowTypePass();
void registerTestFooAnalysisPass();
void registerTestComposeSubView();
void registerTestMultiBuffering();
void registerTestIRVisitorsPass();
void registerTestGenericIRVisitorsPass();
void registerTestInterfaces();
void registerTestIRVisitorsPass();
void registerTestLastModifiedPass();
void registerTestLinalgDecomposeOps();
void registerTestLinalgDropUnitDims();
void registerTestLinalgElementwiseFusion();
void registerTestLinalgGreedyFusion();
void registerTestLinalgRankReduceContractionOps();
void registerTestLinalgTransforms();
void registerTestLivenessAnalysisPass();
void registerTestLivenessPass();
void registerTestLLVMLegalizePatternsPass();
void registerTestLoopFusion();
void registerTestLoopMappingPass();
void registerTestLoopUnrollingPass();
void registerTestLowerToArmSME();
void registerTestLowerToLLVM();
void registerTestMakeIsolatedFromAbovePass();
void registerTestMatchReductionPass();
void registerTestMathAlgebraicSimplificationPass();
void registerTestMathPolynomialApproximationPass();
void registerTestMathToVCIXPass();
void registerTestIrdlTestDialectConversionPass();
void registerTestMemRefDependenceCheck();
void registerTestMemRefStrideCalculation();
void registerTestMemRefToLLVMWithTransforms();
void registerTestReshardingPartitionPass();
void registerTestShardSimplificationsPass();
void registerTestMultiBuffering();
void registerTestNextAccessPass();
void registerTestNVGPULowerings();
void registerTestOpenACC();
void registerTestOneShotModuleBufferizePass();
void registerTestOpaqueLoc();
void registerTestOpLoweringPasses();
void registerTestPadFusion();
void registerTestRecursiveTypesPass();
void registerTestSCFUpliftWhileToFor();
void registerTestSCFUtilsPass();
void registerTestSCFWhileOpBuilderPass();
void registerTestSCFWrapInZeroTripCheckPasses();
void registerTestShapeMappingPass();
void registerTestSingleFold();
void registerTestSliceAnalysisPass();
void registerTestSPIRVCPURunnerPipeline();
void registerTestSPIRVFuncSignatureConversion();
void registerTestSPIRVVectorUnrolling();
void registerTestStridedMetadataRangeAnalysisPass();
void registerTestTensorCopyInsertionPass();
void registerTestTensorLikeAndBufferLikePass();
void registerTestTensorTransforms();
void registerTestTopologicalSortAnalysisPass();
void registerTestTransformDialectEraseSchedulePass();
void registerTestPassStateExtensionCommunication();
void registerTestVectorLowerings();
void registerTestVectorReductionToSPIRVDotProd();
void registerTestVulkanRunnerPipeline();
void registerTestWrittenToPass();
void registerTestXeGPULowerings();
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
void registerTestDialectConversionPasses();
void registerTestPDLByteCodePass();
void registerTestPDLLPasses();
#endif
} // namespace test
} // namespace mlir

namespace test {
void registerTestDialect(DialectRegistry &);
void registerTestDynDialect(DialectRegistry &);
void registerTestTilingInterfaceTransformDialectExtension(DialectRegistry &);
void registerTestTransformDialectExtension(DialectRegistry &);
void registerIrdlTestDialect(DialectRegistry &);
void registerTestTransformsTransformDialectExtension(DialectRegistry &);
} // namespace test

#ifdef MLIR_INCLUDE_TESTS
void registerTestPasses() {
  registerCloneTestPasses();
  registerConvertToTargetEnvPass();
  registerPrintTosaAvailabilityPass();
  registerLazyLoadingTestPasses();
  registerLoopLikeInterfaceTestPasses();
  registerPassManagerTestPass();
  registerPrintSpirvAvailabilityPass();
  registerRegionTestPasses();
  registerShapeFunctionTestPasses();
  registerSideEffectTestPasses();
  registerSliceAnalysisTestPass();
  registerSymbolTestPasses();
  registerTestAffineAccessAnalysisPass();
  registerTestAffineDataCopyPass();
  registerTestAffineLoopUnswitchingPass();
  registerTestAffineReifyValueBoundsPass();
  registerTestAffineWalk();
  registerTestBytecodeRoundtripPasses();
  registerTestDecomposeAffineOpPass();
  registerTestFunc();
  registerTestGpuLoweringPasses();
  registerTestGpuMemoryPromotionPass();
  registerTestLoopPermutationPass();
  registerTestMatchers();
  registerTestOperationEqualPass();
  registerTestPreserveUseListOrders();
  registerTestPrintDefUsePass();
  registerTestPrintInvalidPass();
  registerTestPrintNestingPass();
  registerTestReducer();
  registerTestSpirvEntryPointABIPass();
  registerTestSpirvModuleCombinerPass();
  registerTestTraitsPass();
  registerTosaTestQuantUtilAPIPass();
  registerVectorizerTestPass();

  mlir::test::registerCommutativityUtils();
  mlir::test::registerConvertCallOpPass();
  mlir::test::registerConvertFuncOpPass();
  mlir::test::registerInliner();
  mlir::test::registerInlinerCallback();
  mlir::test::registerMemRefBoundCheck();
  mlir::test::registerPatternsTestPass();
  mlir::test::registerSimpleParametricTilingPass();
  mlir::test::registerTestAffineLoopParametricTilingPass();
  mlir::test::registerTestAliasAnalysisPass();
  mlir::test::registerTestArithEmulateWideIntPass();
  mlir::test::registerTestBuiltinAttributeInterfaces();
  mlir::test::registerTestBuiltinDistinctAttributes();
  mlir::test::registerTestCallGraphPass();
  mlir::test::registerTestCfAssertPass();
  mlir::test::registerTestCFGLoopInfoPass();
  mlir::test::registerTestComposeSubView();
  mlir::test::registerTestCompositePass();
  mlir::test::registerTestControlFlowSink();
  mlir::test::registerTestConvertToSPIRVPass();
  mlir::test::registerTestDataLayoutPropagation();
  mlir::test::registerTestDataLayoutQuery();
  mlir::test::registerTestDeadCodeAnalysisPass();
  mlir::test::registerTestDecomposeCallGraphTypes();
  mlir::test::registerTestDiagnosticsPass();
  mlir::test::registerTestDiagnosticsMetadataPass();
  mlir::test::registerTestDominancePass();
  mlir::test::registerTestDynamicPipelinePass();
  mlir::test::registerTestRemarkPass();
  mlir::test::registerTestEmulateNarrowTypePass();
  mlir::test::registerTestFooAnalysisPass();
  mlir::test::registerTestComposeSubView();
  mlir::test::registerTestMultiBuffering();
  mlir::test::registerTestIRVisitorsPass();
  mlir::test::registerTestGenericIRVisitorsPass();
  mlir::test::registerTestInterfaces();
  mlir::test::registerTestIrdlTestDialectConversionPass();
  mlir::test::registerTestIRVisitorsPass();
  mlir::test::registerTestLastModifiedPass();
  mlir::test::registerTestLinalgDecomposeOps();
  mlir::test::registerTestLinalgDropUnitDims();
  mlir::test::registerTestLinalgElementwiseFusion();
  mlir::test::registerTestLinalgGreedyFusion();
  mlir::test::registerTestLinalgRankReduceContractionOps();
  mlir::test::registerTestLinalgTransforms();
  mlir::test::registerTestLivenessAnalysisPass();
  mlir::test::registerTestLivenessPass();
  mlir::test::registerTestLLVMLegalizePatternsPass();
  mlir::test::registerTestLoopFusion();
  mlir::test::registerTestLoopMappingPass();
  mlir::test::registerTestLoopUnrollingPass();
  mlir::test::registerTestLowerToArmSME();
  mlir::test::registerTestLowerToLLVM();
  mlir::test::registerTestMakeIsolatedFromAbovePass();
  mlir::test::registerTestMatchReductionPass();
  mlir::test::registerTestMathAlgebraicSimplificationPass();
  mlir::test::registerTestMathPolynomialApproximationPass();
  mlir::test::registerTestMathToVCIXPass();
  mlir::test::registerTestMemRefDependenceCheck();
  mlir::test::registerTestMemRefStrideCalculation();
  mlir::test::registerTestMemRefToLLVMWithTransforms();
  mlir::test::registerTestReshardingPartitionPass();
  mlir::test::registerTestShardSimplificationsPass();
  mlir::test::registerTestMultiBuffering();
  mlir::test::registerTestNextAccessPass();
  mlir::test::registerTestNVGPULowerings();
  mlir::test::registerTestOpenACC();
  mlir::test::registerTestOneShotModuleBufferizePass();
  mlir::test::registerTestOpaqueLoc();
  mlir::test::registerTestOpLoweringPasses();
  mlir::test::registerTestPadFusion();
  mlir::test::registerTestRecursiveTypesPass();
  mlir::test::registerTestSCFUpliftWhileToFor();
  mlir::test::registerTestSCFUtilsPass();
  mlir::test::registerTestSCFWhileOpBuilderPass();
  mlir::test::registerTestSCFWrapInZeroTripCheckPasses();
  mlir::test::registerTestShapeMappingPass();
  mlir::test::registerTestSingleFold();
  mlir::test::registerTestSliceAnalysisPass();
  mlir::test::registerTestSPIRVCPURunnerPipeline();
  mlir::test::registerTestSPIRVFuncSignatureConversion();
  mlir::test::registerTestSPIRVVectorUnrolling();
  mlir::test::registerTestStridedMetadataRangeAnalysisPass();
  mlir::test::registerTestTensorCopyInsertionPass();
  mlir::test::registerTestTensorLikeAndBufferLikePass();
  mlir::test::registerTestTensorTransforms();
  mlir::test::registerTestTopologicalSortAnalysisPass();
  mlir::test::registerTestTransformDialectEraseSchedulePass();
  mlir::test::registerTestPassStateExtensionCommunication();
  mlir::test::registerTestVectorLowerings();
  mlir::test::registerTestVectorReductionToSPIRVDotProd();
  mlir::test::registerTestVulkanRunnerPipeline();
  mlir::test::registerTestWrittenToPass();
  mlir::test::registerTestXeGPULowerings();
#if MLIR_ENABLE_PDL_IN_PATTERNMATCH
  mlir::test::registerTestDialectConversionPasses();
  mlir::test::registerTestPDLByteCodePass();
  mlir::test::registerTestPDLLPasses();
#endif
}
#endif

int main(int argc, char **argv) {
  registerAllPasses();
#ifdef MLIR_INCLUDE_TESTS
  registerTestPasses();
#endif
  DialectRegistry registry;
  registerAllDialects(registry);
  registerAllExtensions(registry);

  // TODO: Remove this and the corresponding MLIRToLLVMIRTranslationRegistration
  // cmake dependency when a safe dialect interface registration mechanism is
  // implemented, see D157703 (and corresponding note on the declaration).
  registerAllGPUToLLVMIRTranslations(registry);

#ifdef MLIR_INCLUDE_TESTS
  ::test::registerIrdlTestDialect(registry);
  ::test::registerTestDialect(registry);
  ::test::registerTestDynDialect(registry);
  ::test::registerTestTilingInterfaceTransformDialectExtension(registry);
  ::test::registerTestTransformDialectExtension(registry);
  ::test::registerTestTransformsTransformDialectExtension(registry);
#endif
  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "MLIR modular optimizer driver\n", registry));
}
