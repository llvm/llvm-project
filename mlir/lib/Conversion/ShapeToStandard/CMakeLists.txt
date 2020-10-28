add_mlir_conversion_library(MLIRShapeToStandard
  ShapeToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ShapeToStandard

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREDSC
  MLIRIR
  MLIRShape
  MLIRPass
  MLIRSCF
  MLIRTransforms
  )
