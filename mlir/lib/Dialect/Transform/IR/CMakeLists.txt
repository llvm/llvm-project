add_mlir_dialect_library(MLIRTransformDialect
  TransformDialect.cpp
  TransformInterfaces.cpp
  TransformOps.cpp

  DEPENDS
  MLIRTransformDialectIncGen
  MLIRTransformInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPDL
  MLIRPDLInterp
  MLIRRewrite
  MLIRSideEffectInterfaces
  )
