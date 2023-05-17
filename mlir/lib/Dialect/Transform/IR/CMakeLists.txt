add_mlir_dialect_library(MLIRTransformDialect
  MatchInterfaces.cpp
  TransformAttrs.cpp
  TransformDialect.cpp
  TransformInterfaces.cpp
  TransformOps.cpp
  TransformTypes.cpp

  DEPENDS
  MLIRMatchInterfacesIncGen
  MLIRTransformDialectIncGen
  MLIRTransformInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRParser
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRRewrite
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformDialectUtils
  )
