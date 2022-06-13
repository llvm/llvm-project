add_mlir_dialect_library(MLIRAMXDialect
  AMXDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/AMX

  DEPENDS
  MLIRAMXIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
