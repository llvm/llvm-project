add_mlir_conversion_library(MLIRMathToLLVMSPV
  MathToLLVMSPV.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/MathToLLVMSPV

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
  MLIRLLVMDialect
  MLIRPass
  MLIRTransformUtils
  MLIRVectorDialect
  MLIRVectorUtils
  )
