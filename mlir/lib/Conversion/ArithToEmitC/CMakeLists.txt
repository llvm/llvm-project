add_mlir_conversion_library(MLIRArithToEmitC
  ArithToEmitC.cpp
  ArithToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/ArithToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIREmitCDialect
  MLIRPass
  MLIRTransformUtils
  )
