add_mlir_dialect_library(MLIRControlFlow
  ControlFlowOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ControlFlow/IR

  DEPENDS
  MLIRControlFlowOpsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRControlFlowInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  )
