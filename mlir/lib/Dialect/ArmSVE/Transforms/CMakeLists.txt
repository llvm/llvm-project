add_mlir_dialect_library(MLIRArmSVETransforms
  LegalizeForLLVMExport.cpp
  LegalizeVectorStorage.cpp

  DEPENDS
  MLIRArmSVEConversionsIncGen
  MLIRArmSVEPassIncGen

  LINK_LIBS PUBLIC
  MLIRArmSVEDialect
  MLIRFuncDialect
  MLIRVectorDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
