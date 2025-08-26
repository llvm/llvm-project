add_mlir_dialect_library(MLIRAMXTransforms
  LegalizeForLLVMExport.cpp

  LINK_LIBS PUBLIC
  MLIRAMXDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
