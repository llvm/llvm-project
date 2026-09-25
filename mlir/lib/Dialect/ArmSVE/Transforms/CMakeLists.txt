add_mlir_dialect_library(MLIRArmSVETransforms
  LegalizeForLLVMExport.cpp
  LegalizeVectorStorage.cpp
  LowerContractToSVEPatterns.cpp

  DEPENDS
  MLIRArmSVEPassIncGen

  LINK_LIBS PUBLIC
  MLIRArmSVEDialect
  MLIRFuncDialect
  MLIRVectorDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
