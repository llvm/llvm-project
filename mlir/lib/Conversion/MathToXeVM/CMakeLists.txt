add_mlir_conversion_library(MLIRMathToXeVM
  MathToXeVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToXeVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRMathDialect
  MLIRLLVMCommonConversion
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  )
