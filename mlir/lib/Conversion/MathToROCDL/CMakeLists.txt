add_mlir_conversion_library(MLIRMathToROCDL
  MathToROCDL.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToROCDL

  DEPENDS
  MLIRConversionPassIncGen

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRDialectUtils
  MLIRFuncDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRMathDialect
  MLIRLLVMCommonConversion
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorUtils
  )
