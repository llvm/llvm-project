add_mlir_dialect_library(MLIRAMDGPUTransforms
  EmulateAtomics.cpp
  OptimizeSharedMemory.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AMDGPU/Transforms

  DEPENDS
  MLIRAMDGPUTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAMDGPUUtils
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorDialect
  )
