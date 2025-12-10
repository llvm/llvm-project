add_mlir_conversion_library(MLIRConvertToEmitC
  ConvertToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ConvertToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToEmitC
  MLIRFuncToEmitC
  MLIRMemRefToEmitC
  MLIRPass
  MLIRSCFToEmitC
  MLIRTransformUtils
  )
