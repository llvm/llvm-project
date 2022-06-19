add_mlir_conversion_library(MLIRBufferizationToMemRef
  BufferizationToMemRef.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/BufferizationToMemRef

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRBufferizationDialect
  MLIRPass
  MLIRTransforms
  )
