add_mlir_conversion_library(MLIRAffineToStandard
  AffineToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/AffineToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRAffine
  MLIRArithmetic
  MLIRMemRef
  MLIRSCF
  MLIRPass
  MLIRStandard
  MLIRTransforms
  MLIRIR
  )
