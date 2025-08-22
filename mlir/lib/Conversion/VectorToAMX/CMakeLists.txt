add_mlir_conversion_library(MLIRVectorToAMX
  VectorToAMX.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToAMX

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAMXDialect
  MLIRAffineUtils
  MLIRArithDialect
  MLIRLinalgUtils
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRTransforms
  MLIRVectorDialect
  )
