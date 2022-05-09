add_mlir_dialect_library(MLIRNVGPU
  NVGPUDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/NVGPU

  DEPENDS
  MLIRNVGPUIncGen

  LINK_LIBS PUBLIC
  MLIRGPUOps
  MLIRIR
  MLIRSideEffectInterfaces
  )
