add_mlir_dialect_library(MLIRAMDGPUDialect
  AMDGPUDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AMDGPU

  DEPENDS
  MLIRAMDGPUEnumsGen
  MLIRAMDGPUAttributesIncGen
  MLIRAMDGPUIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSideEffectInterfaces
  )
