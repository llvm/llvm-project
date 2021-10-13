add_mlir_conversion_library(MLIRComplexToStandard
  ComplexToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ComplexToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRComplex
  MLIRIR
  MLIRMath
  MLIRStandard
  MLIRTransforms
  )
