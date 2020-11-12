# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestAffineLoopParametricTiling.cpp
  TestExpandTanh.cpp
  TestCallGraph.cpp
  TestConstantFold.cpp
  TestConvVectorization.cpp
  TestConvertCallOp.cpp
  TestConvertGPUKernelToCubin.cpp
  TestConvertGPUKernelToHsaco.cpp
  TestDominance.cpp
  TestDynamicPipeline.cpp
  TestFinalizingBufferize.cpp
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
  TestMemRefBoundCheck.cpp
  TestMemRefDependenceCheck.cpp
  TestMemRefStrideCalculation.cpp
  TestSCFUtils.cpp
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
  MLIRGPUToGPURuntimeTransforms
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRNVVMIR
  MLIRSCF
  MLIRSCFTransforms
  MLIRGPU
  MLIRPass
  MLIRROCDLIR
  MLIRStandardOpsTransforms
  MLIRTargetNVVMIR
  MLIRTargetROCDLIR
  MLIRTestDialect
  MLIRTransformUtils
  MLIRVectorToSCF
  MLIRVector
  )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../Dialect/Test)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../Dialect/Test)
