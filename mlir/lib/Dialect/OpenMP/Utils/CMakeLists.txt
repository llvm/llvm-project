add_mlir_dialect_library(MLIROpenMPUtils
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/OpenMP

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIROpenACCMPCommon
  MLIROpenMPDialect
  MLIRSupport
  )
