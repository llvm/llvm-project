add_mlir_conversion_library(MLIRFuncToEmitC
  FuncToEmitC.cpp
  FuncToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/FuncToEmitC

  LINK_LIBS PUBLIC
  MLIREmitCCommonConversion
  MLIREmitCDialect
  MLIRFuncDialect
  MLIRPass
  MLIRTransformUtils
  )
