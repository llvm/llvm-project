add_mlir_dialect_library(MLIRTransformTuneExtension
  TuneExtension.cpp
  TuneExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectTuneExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
)
