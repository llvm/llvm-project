add_mlir_dialect_library(MLIRTransformDialect
  TransformAttrs.cpp
  TransformDialect.cpp
  TransformOps.cpp
  TransformTypes.cpp
  Utils.cpp

  DEPENDS
  MLIRTransformDialectIncGen
  MLIRTransformOpsIncGen
  MLIRTransformTypesIncGen

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
