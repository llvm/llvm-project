add_mlir_dialect_library(MLIROpenACCDialect
  OpenACC.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenACC

  DEPENDS
  MLIROpenACCOpsIncGen
  MLIROpenACCEnumsIncGen
  MLIROpenACCAttributesIncGen
  MLIROpenACCMPOpsInterfacesIncGen
  MLIROpenACCTypeInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIROpenACCMPCommon
  )

