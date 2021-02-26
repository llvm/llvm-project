# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestAffineLoopParametricTiling.cpp
  TestExpandTanh.cpp
  TestCallGraph.cpp
  TestDecomposeCallGraphTypes.cpp
  TestConstantFold.cpp
  TestConvVectorization.cpp
  TestConvertCallOp.cpp
  TestConvertGPUKernelToCubin.cpp
  TestConvertGPUKernelToHsaco.cpp
  TestDominance.cpp
  TestDynamicPipeline.cpp
  TestLoopFusion.cpp
  TestGpuMemoryPromotion.cpp
  TestGpuParallelLoopMapping.cpp
  TestGpuRewrite.cpp
  TestInlining.cpp
  TestLinalgCodegenStrategy.cpp
  TestLinalgFusionTransforms.cpp
  TestLinalgHoisting.cpp
  TestLinalgTransforms.cpp
  TestLiveness.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestNumberOfExecutions.cpp
  TestOpaqueLoc.cpp
  TestPolynomialApproximation.cpp
  TestMemRefBoundCheck.cpp
  TestMemRefDependenceCheck.cpp
  TestMemRefStrideCalculation.cpp
  TestSCFUtils.cpp
  TestSparsification.cpp
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRAnalysis
  MLIREDSC
  MLIRGPU
  MLIRGPU
  MLIRGPUToGPURuntimeTransforms
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRMathTransforms
  MLIRNVVMIR
  MLIRNVVMToLLVMIRTranslation
  MLIRPass
  MLIRROCDLIR
  MLIRROCDLToLLVMIRTranslation
  MLIRSCF
  MLIRSCFTransforms
  MLIRStandardOpsTransforms
  MLIRTargetLLVMIR
  MLIRTestDialect
  MLIRTransformUtils
  MLIRVector
  MLIRVectorToSCF
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
