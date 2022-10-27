add_mlir_dialect_library(MLIRVectorDialect
  VectorOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Vector/IR

  DEPENDS
  MLIRVectorOpsIncGen
  MLIRVectorOpsEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowInterfaces
  MLIRDataLayoutInterfaces
  MLIRDestinationStyleOpInterface
  MLIRDialectUtils
  MLIRIR
  MLIRMaskingInterfaces
  MLIRMemRefDialect
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRVectorInterfaces
  )
