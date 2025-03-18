add_mlir_conversion_library(MLIRVectorToAMDGPU
  VectorToAMDGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToAMDGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )
