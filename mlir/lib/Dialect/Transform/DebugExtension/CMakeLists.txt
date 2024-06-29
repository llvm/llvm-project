add_mlir_dialect_library(MLIRTransformDebugExtension
  DebugExtension.cpp
  DebugExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectDebugExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
)
