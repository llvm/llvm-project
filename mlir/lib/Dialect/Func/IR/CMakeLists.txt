add_mlir_dialect_library(MLIRFuncDialect
  FuncOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Func

  DEPENDS
  MLIRFuncOpsIncGen

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRControlFlowDialect
  MLIRControlFlowInterfaces
  MLIRInferTypeOpInterface
  MLIRIR
  MLIRSideEffectInterfaces
  )
