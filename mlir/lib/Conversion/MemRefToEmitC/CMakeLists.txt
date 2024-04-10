add_mlir_conversion_library(MLIRMemRefToEmitC
  MemRefToEmitC.cpp
  MemRefToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MemRefToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRMemRefDialect
  MLIRTransforms
  )
