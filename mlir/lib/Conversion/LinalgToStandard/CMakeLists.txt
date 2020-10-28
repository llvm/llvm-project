add_mlir_conversion_library(MLIRLinalgToStandard
  LinalgToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRLinalg
  MLIRPass
  MLIRSCF
  MLIRTransforms
  )
