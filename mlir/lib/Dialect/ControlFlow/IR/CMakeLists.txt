add_mlir_dialect_library(MLIRControlFlowDialect
  ControlFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR

  DEPENDS
  MLIRControlFlowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  )
