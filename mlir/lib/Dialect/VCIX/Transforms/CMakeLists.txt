add_mlir_dialect_library(MLIRVCIXTransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRVCIXConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRVCIXDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
