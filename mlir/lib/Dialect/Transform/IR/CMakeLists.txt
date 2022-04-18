add_mlir_dialect_library(MLIRTransformDialect
  TransformDialect.cpp
  TransformInterfaces.cpp

  DEPENDS
  MLIRTransformDialectIncGen
  MLIRTransformInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
