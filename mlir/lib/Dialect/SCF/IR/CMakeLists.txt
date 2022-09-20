add_mlir_dialect_library(MLIRSCFDialect
  SCF.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/SCF

  DEPENDS
  MLIRSCFOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithmeticDialect
  MLIRBufferizationDialect
  MLIRControlFlowDialect
  MLIRIR
  MLIRLoopLikeInterface
  MLIRSideEffectInterfaces
  )

