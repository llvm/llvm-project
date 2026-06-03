add_mlir_dialect_library(MLIREmitCTransforms
  Transforms.cpp
  FormExpressions.cpp
  MLGOScalarizeFunctionResult.cpp
  TypeConversions.cpp
  WrapFuncInClass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/EmitC/Transforms

  DEPENDS
  MLIREmitCTransformsIncGen

  LINK_LIBS PUBLIC
  # MLGO related
  MLIRArithDialect
  # MLGO related
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIREmitCDialect
  # MLGO related
  MLIRTensorDialect
  MLIRTransforms
)
