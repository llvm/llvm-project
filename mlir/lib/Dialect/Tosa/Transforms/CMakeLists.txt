add_mlir_dialect_library(MLIRTosaTransforms
  TosaAttachTarget.cpp
  TosaArithConstantToConst.cpp
  TosaConvertIntegerTypeToSignless.cpp
  TosaDecomposeTransposeConv.cpp
  TosaDecomposeDepthwise.cpp
  TosaFolders.cpp
  TosaInferShapes.cpp
  TosaLayerwiseConstantFoldPass.cpp
  TosaMakeBroadcastable.cpp
  TosaOptionalDecompositions.cpp
  TosaReduceTransposes.cpp
  TosaTypeConverters.cpp
  TosaProfileCompliance.cpp
  TosaValidation.cpp
  TosaNarrowTypes.cpp
  TosaInputShape.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Tosa/Transforms

  DEPENDS
  MLIRTosaPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRFuncTransformOps
  MLIRPass
  MLIRTosaDialect
  MLIRTransformUtils
  MLIRFuncTransforms
  )
