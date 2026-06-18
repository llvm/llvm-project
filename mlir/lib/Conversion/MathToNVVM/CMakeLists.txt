add_mlir_conversion_library(MLIRMathToNVVM
  MathToNVVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToNVVM

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRMathDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorUtils
  )
