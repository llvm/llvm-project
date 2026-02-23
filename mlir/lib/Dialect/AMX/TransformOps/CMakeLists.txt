add_mlir_dialect_library(MLIRAMXTransformOps
  AMXTransformOps.cpp

  DEPENDS
  MLIRAMXTransformOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRVectorDialect
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRTransformDialectUtils
  MLIRAMXDialect
  MLIRAMXTransforms
  )
