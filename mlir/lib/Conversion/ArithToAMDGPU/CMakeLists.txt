add_mlir_conversion_library(MLIRArithToAMDGPU
  ArithToAMDGPU.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToAMDGPU

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAMDGPUDialect
  MLIRAMDGPUUtils
  MLIRArithDialect
  MLIRArithUtils
  MLIRVectorDialect
  MLIRPass
  MLIRTransforms
  )
