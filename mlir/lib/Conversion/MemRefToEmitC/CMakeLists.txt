add_mlir_conversion_library(MLIRMemRefToEmitC
  MemRefToEmitC.cpp
  MemRefToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefToEmitC

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREmitCCommonConversion
  MLIREmitCDialect
  MLIRMemRefDialect
  MLIRTransforms
  )
