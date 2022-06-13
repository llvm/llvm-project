set(LLVM_TARGET_DEFINITIONS ShapeCanonicalization.td)
mlir_tablegen(ShapeCanonicalization.inc -gen-rewriters)
add_public_tablegen_target(MLIRShapeCanonicalizationIncGen)

add_mlir_dialect_library(MLIRShapeDialect
  Shape.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Shape

  DEPENDS
  MLIRShapeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRFuncDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  )
