set(LLVM_TARGET_DEFINITIONS ShapeToStandard.td)
mlir_tablegen(ShapeToStandard.cpp.inc -gen-rewriters)
add_public_tablegen_target(ShapeToStandardIncGen)

add_mlir_conversion_library(MLIRShapeToStandard
  ConvertShapeConstraints.cpp
  ShapeToStandard.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ShapeToStandard

  DEPENDS
  MLIRConversionPassIncGen
  ShapeToStandardIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRShapeDialect
  MLIRTensorDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  )
