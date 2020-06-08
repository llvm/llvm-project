add_mlir_dialect_library(MLIRShape
  IR/Shape.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Shape

  DEPENDS
  MLIRShapeOpsIncGen

  LINK_LIBS PUBLIC
  MLIRControlFlowInterfaces
  MLIRDialect
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  )
