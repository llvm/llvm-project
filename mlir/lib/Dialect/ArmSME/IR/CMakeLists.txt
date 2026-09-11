add_mlir_dialect_library(MLIRArmSMEDialect
  ArmSME.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME

  DEPENDS
  MLIRArmSMEOpsIncGen
  MLIRArmSMEIntrinsicOpsIncGen
  MLIRArmSMEIncGen
  MLIRArmSMEConversionsIncGen
  MLIRArmSMEOpInterfaces

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSideEffectInterfaces
  MLIRVectorDialect
)
