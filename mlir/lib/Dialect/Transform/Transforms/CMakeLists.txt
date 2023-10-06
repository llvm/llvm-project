add_mlir_dialect_library(MLIRTransformDialectTransforms
  CheckUses.cpp
  InferEffects.cpp
  TransformInterpreterPassBase.cpp
  TransformInterpreterUtils.cpp

  DEPENDS
  MLIRTransformDialectTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRFunctionInterfaces
  MLIRTransformDialect
  MLIRIR
  MLIRPass
)
