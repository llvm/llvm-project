add_mlir_dialect_library(MLIRNVGPUDialect
  NVGPUDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/NVGPU

  DEPENDS
  MLIRNVGPUIncGen
  MLIRNVGPUEnumsIncGen
  MLIRNVGPUAttributesIncGen
  MLIRNVGPUTypesIncGen

  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRIR
  MLIRSideEffectInterfaces
  )
