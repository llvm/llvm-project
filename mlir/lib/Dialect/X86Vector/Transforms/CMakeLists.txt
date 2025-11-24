add_mlir_dialect_library(MLIRX86VectorTransforms
  AVXTranspose.cpp
  LegalizeForLLVMExport.cpp
  VectorContractToFMA.cpp
  VectorContractToPackedTypeDotProduct.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRX86VectorDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRVectorUtils
  )
