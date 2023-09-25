add_mlir_dialect_library(MLIRArmSMEDialect
  ArmSME.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME

  DEPENDS
  MLIRArmSMEIncGen
  MLIRArmSMEAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRVectorDialect
)
