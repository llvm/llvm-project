add_mlir_conversion_library(MLIRShapeToSCF
  ShapeToSCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ShapeToSCF

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRShape
  MLIRPass
  MLIRSCF
  MLIRTransforms
  )
