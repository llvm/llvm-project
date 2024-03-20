add_mlir_library(MLIRTransformDialectInterfaces
  TransformInterfaces.cpp

  DEPENDS
  MLIRTransformInterfacesIncGen

  LINK_LIBS PUBLIC
  MLIRCastInterfaces
  MLIRIR
  MLIRRewrite
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformDialectUtils
)

