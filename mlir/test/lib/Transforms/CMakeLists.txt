# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestAllReduceLowering.cpp
  TestBufferPlacement.cpp
  TestExpandTanh.cpp
  TestCallGraph.cpp
  TestConstantFold.cpp
  TestConvertGPUKernelToCubin.cpp
  TestConvertGPUKernelToHsaco.cpp
  TestDominance.cpp
  TestLoopFusion.cpp
  TestGpuMemoryPromotion.cpp
  TestGpuParallelLoopMapping.cpp
  TestInlining.cpp
  TestLinalgHoisting.cpp
  TestLinalgTransforms.cpp
  TestLiveness.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
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
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRGPU
  MLIRGPUToGPURuntimeTransforms
  MLIRLinalgOps
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
