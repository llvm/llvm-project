add_mlir_dialect_library(MLIRArmSVETransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRArmSVEConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRArmSVEDialect
  MLIRFuncDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
