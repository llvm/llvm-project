add_mlir_dialect_library(MLIRTransformSMTExtension
  SMTExtension.cpp
  SMTExtensionOps.cpp

  DEPENDS
  MLIRTransformDialectSMTExtensionOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRTransformDialect
  MLIRSMT
)
