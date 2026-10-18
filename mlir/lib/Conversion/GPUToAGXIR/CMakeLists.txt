add_mlir_conversion_library(MLIRGPUToAGXIRTransforms
  LowerGpuOpsToAGXIROps.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRAGXDialect
  MLIRAGXIRDialect
  MLIRAGXToAGXIR
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRPass
  )
