if (MLIR_CUDA_CONVERSIONS_ENABLED)
  set(NVPTX_LIBS
    MC
    NVPTXCodeGen
    NVPTXDesc
    NVPTXInfo
  )
endif()

if (MLIR_ROCM_CONVERSIONS_ENABLED)
  set(AMDGPU_LIBS
    MC
    AMDGPUCodeGen
    AMDGPUDesc
    AMDGPUInfo
  )
endif()

# Exclude tests from libMLIR.so
add_mlir_library(MLIRTestTransforms
  TestAllReduceLowering.cpp
  TestBufferPlacement.cpp
  TestCallGraph.cpp
  TestConstantFold.cpp
  TestConvertGPUKernelToCubin.cpp
  TestConvertGPUKernelToHsaco.cpp
  TestDominance.cpp
  TestLoopFusion.cpp
  TestGpuMemoryPromotion.cpp
  TestGpuParallelLoopMapping.cpp
  TestInlining.cpp
  TestLinalgTransforms.cpp
  TestLiveness.cpp
  TestLoopMapping.cpp
  TestLoopParametricTiling.cpp
  TestLoopUnrolling.cpp
  TestOpaqueLoc.cpp
  TestMemRefBoundCheck.cpp
  TestMemRefDependenceCheck.cpp
  TestMemRefStrideCalculation.cpp
  TestVectorToSCFConversion.cpp
  TestVectorTransforms.cpp

  EXCLUDE_FROM_LIBMLIR

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Transforms

  DEPENDS
  MLIRStandardOpsIncGen
  MLIRTestVectorTransformPatternsIncGen

  LINK_COMPONENTS
  ${AMDGPU_LIBS}
  ${NVPTX_LIBS}

  LINK_LIBS PUBLIC
  MLIRAffineOps
  MLIRAnalysis
  MLIREDSC
  MLIRGPU
  MLIRGPUtoGPURuntimeTransforms
  MLIRLinalgOps
  MLIRLinalgTransforms
  MLIRNVVMIR
  MLIRSCF
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../DeclarativeTransforms)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../DeclarativeTransforms)
