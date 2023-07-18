add_mlir_dialect_library(MLIRNVGPUTransforms
  CreateAsyncGroups.cpp
  OptimizeSharedMemory.cpp
  MmaSyncTF32Transform.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/NVGPU

  DEPENDS
  MLIRNVGPUPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRPass
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorUtils
)
