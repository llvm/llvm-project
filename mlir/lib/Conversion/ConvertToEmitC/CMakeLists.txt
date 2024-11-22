set(LLVM_OPTIONAL_SOURCES
  ConvertToEmitCPass.cpp
)

add_mlir_conversion_library(MLIRConvertToEmitCPass
  ConvertToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ConvertToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithToEmitC
  MLIRFuncToEmitC
  MLIRIR
  MLIRMemRefToEmitC
  MLIRPass
  MLIRRewrite
  MLIRSCFToEmitC
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  )
