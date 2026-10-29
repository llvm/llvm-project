add_mlir_dialect_library(MLIRX86TransformOps
  X86TransformOps.cpp

  DEPENDS
  MLIRX86TransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRVectorDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRX86Dialect
  MLIRX86Transforms
  )
