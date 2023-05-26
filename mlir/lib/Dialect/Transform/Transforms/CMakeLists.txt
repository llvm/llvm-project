add_mlir_dialect_library(MLIRTransformDialectTransforms
  CheckUses.cpp
  InferEffects.cpp
  TransformInterpreterPassBase.cpp

  DEPENDS
  MLIRTransformDialectTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTransformDialect
  MLIRIR
  MLIRPass
)
