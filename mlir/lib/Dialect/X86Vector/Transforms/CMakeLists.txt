add_mlir_dialect_library(MLIRX86VectorTransforms
  AVXTranspose.cpp
  LegalizeForLLVMExport.cpp
  VectorContractToFMA.cpp
  VectorContractToPackedTypeDotProduct.cpp
  VectorContractBF16ToFMA.cpp
  SinkVectorProducerOps.cpp
  ShuffleVectorFMAOps.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRLinalgDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRVectorUtils
  MLIRX86VectorDialect
  MLIRX86VectorUtils
  )
