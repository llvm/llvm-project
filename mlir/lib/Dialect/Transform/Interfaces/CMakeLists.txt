add_mlir_library(MLIRTransformDialectInterfaces
  MatchInterfaces.cpp
  TransformInterfaces.cpp

  DEPENDS
  MLIRMatchInterfacesIncGen
  MLIRTransformInterfacesIncGen
  MLIRTransformDialectTypeInterfacesIncGen
  MLIRTransformDialectAttrInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRCastInterfaces
  MLIRIR
  MLIRRewrite
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformDialectUtils
)

