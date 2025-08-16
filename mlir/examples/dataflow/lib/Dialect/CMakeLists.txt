add_mlir_dialect_library(MLIRStringDialect
  StringDialect.cpp

  DEPENDS
  MLIRStringOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
)
