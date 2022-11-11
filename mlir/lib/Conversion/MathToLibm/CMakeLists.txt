add_mlir_conversion_library(MLIRMathToLibm
  MathToLibm.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToLibm

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRMathDialect
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorUtils
  )
