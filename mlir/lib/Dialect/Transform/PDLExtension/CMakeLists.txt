add_mlir_dialect_library(MLIRTransformPDLExtension
  PDLExtension.cpp
  PDLExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectPDLExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
  MLIRPDLDialect
  MLIRPDLInterpDialect
)
