add_mlir_dialect_library(MLIRArmNeonTransforms
  LowerContractionToSMMLAPattern.cpp

  DEPENDS
  MLIRArmNeonIncGen

  LINK_LIBS PUBLIC
  MLIRArmNeonDialect
  MLIRFuncDialect
  MLIRVectorDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  )
