add_mlir_dialect_library(MLIRX86Dialect
  X86Dialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/X86

  DEPENDS
  MLIRX86IncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRSideEffectInterfaces
  )
