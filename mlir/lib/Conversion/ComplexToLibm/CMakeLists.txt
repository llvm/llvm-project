add_mlir_conversion_library(MLIRComplexToLibm
  ComplexToLibm.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ComplexToLibm

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRFunc
  MLIRComplex
  MLIRTransformUtils
  )
