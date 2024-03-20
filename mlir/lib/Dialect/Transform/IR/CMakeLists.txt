add_mlir_dialect_library(MLIRTransformDialect
  MatchInterfaces.cpp
  TransformAttrs.cpp
  TransformDialect.cpp
  TransformOps.cpp
  TransformTypes.cpp
  Utils.cpp

  DEPENDS
  MLIRMatchInterfacesIncGen
  MLIRTransformDialectIncGen

  LINK_LIBS PUBLIC
  MLIRCastInterfaces
  MLIRFunctionInterfaces
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRLoopLikeInterface
  MLIRParser
  MLIRPass
  MLIRRewrite
  MLIRSideEffectInterfaces
  MLIRTransforms
  MLIRTransformDialectInterfaces
  MLIRTransformDialectUtils
  )
