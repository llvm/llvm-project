add_mlir_conversion_library(MLIRVectorToGPUCommonConversion
  VectorToGPUUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRGPUDialect
  MLIRIR
  MLIRSCFDialect
  MLIRVectorDialect
  )
