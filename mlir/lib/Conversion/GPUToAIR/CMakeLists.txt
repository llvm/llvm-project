add_mlir_conversion_library(MLIRGPUToAIRTransforms
  LowerGpuOpsToAIROps.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAIRDialect
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMathDialect
  MLIRMemRefToLLVM
  MLIRPass
  MLIRSCFDialect
  MLIRSCFToControlFlow
  MLIRVectorDialect
  )
