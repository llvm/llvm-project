add_mlir_dialect_library(MLIRTransformLoopExtension
  LoopExtension.cpp
  LoopExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectLoopExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLoopLikeInterface
  MLIRTransformDialect
  MLIRTransforms
)
