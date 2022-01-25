add_mlir_dialect_library(MLIRArmSVETransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRArmSVEConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRArmSVE
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  MLIRStandard
  )
