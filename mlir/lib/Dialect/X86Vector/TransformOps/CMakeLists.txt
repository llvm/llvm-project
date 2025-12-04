add_mlir_dialect_library(MLIRX86VectorTransformOps
  X86VectorTransformOps.cpp

  DEPENDS
  MLIRX86VectorTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRX86VectorDialect
  MLIRX86VectorTransforms
  )
