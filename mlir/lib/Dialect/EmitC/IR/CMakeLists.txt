add_mlir_dialect_library(MLIREmitCDialect
  EmitC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/EmitC

  DEPENDS
  MLIREmitCIncGen
  MLIREmitCAttributesIncGen

  LINK_LIBS PUBLIC
  MLIRCallInterfaces
  MLIRCastInterfaces
  MLIRControlFlowInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRSideEffectInterfaces
  )
