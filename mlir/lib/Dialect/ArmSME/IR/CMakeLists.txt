add_mlir_dialect_library(MLIRArmSMEDialect
  ArmSMEDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/ArmSME

  DEPENDS
  MLIRArmSMEIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  )
