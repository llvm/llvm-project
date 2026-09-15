add_mlir_conversion_library(MLIRMathToEmitC
  MathToEmitC.cpp
  MathToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToEmitC

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIREmitCDialect
  MLIRMathDialect
  MLIRPass
  MLIRTransformUtils
)
