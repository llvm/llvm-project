add_mlir_conversion_library(MLIRSCFToEmitC
  SCFToEmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/SCFToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIREmitCDialect
  MLIREmitCTransforms
  MLIRSCFDialect
  MLIRTransforms
  )
