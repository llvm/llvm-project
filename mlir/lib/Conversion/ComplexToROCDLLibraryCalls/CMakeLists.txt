add_mlir_conversion_library(MLIRComplexToROCDLLibraryCalls
  ComplexToROCDLLibraryCalls.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ComplexToROCDLLibraryCalls

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRComplexDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
  )
