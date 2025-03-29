add_mlir_library(MLIRTransformDialectInterfaces
  MatchInterfaces.cpp
  TransformInterfaces.cpp

  DEPENDS
  MLIRMatchInterfacesIncGen
  MLIRTransformInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRCastInterfaces
  MLIRIR
  MLIRRewrite
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformDialectUtils
)

