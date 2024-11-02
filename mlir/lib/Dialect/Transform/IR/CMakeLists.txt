add_mlir_dialect_library(MLIRTransformDialect
  TransformDialect.cpp
  TransformInterfaces.cpp
  TransformOps.cpp
  TransformTypes.cpp

  DEPENDS
  MLIRTransformDialectIncGen
  MLIRTransformInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRRewrite
  MLIRSideEffectInterfaces
  )
