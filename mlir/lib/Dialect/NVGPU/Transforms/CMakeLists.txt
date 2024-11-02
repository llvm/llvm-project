add_mlir_dialect_library(MLIRNVGPUTransforms
  OptimizeSharedMemory.cpp
  MmaSyncTF32Transform.cpp  

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/NVGPU

  DEPENDS
  MLIRNVGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUOps
  MLIRIR
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorUtils
)
