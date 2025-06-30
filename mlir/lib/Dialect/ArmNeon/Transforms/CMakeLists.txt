add_mlir_dialect_library(MLIRArmNeonTransforms
  LowerContractionToNeonI8MMPattern.cpp

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
