add_mlir_dialect_library(MLIRArmSMEDialect
  ArmSME.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME

  DEPENDS
  MLIRArmSMEAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRVectorDialect
)
