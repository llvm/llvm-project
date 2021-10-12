add_mlir_dialect_library(MLIRX86VectorTransforms
  LegalizeForLLVMExport.cpp

  DEPENDS
  MLIRX86VectorConversionsIncGen

  LINK_LIBS PUBLIC
  MLIRArithmetic
  MLIRX86Vector
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMIR
  )
