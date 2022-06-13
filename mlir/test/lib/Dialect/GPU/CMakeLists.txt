# Exclude tests from libMLIR.so
add_mlir_library(MLIRGPUTestPasses
  TestConvertGPUKernelToCubin.cpp
  TestConvertGPUKernelToHsaco.cpp
  TestGpuMemoryPromotion.cpp
  TestGpuRewrite.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRGPUOps
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMToLLVMIRTranslation
  MLIRNVVMDialect
  MLIRNVVMToLLVMIRTranslation
  MLIRPass
  MLIRROCDLDialect
  MLIRROCDLToLLVMIRTranslation
  MLIRSCFDialect
  MLIRTransformUtils
  )
