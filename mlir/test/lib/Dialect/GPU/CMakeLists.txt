# Exclude tests from libMLIR.so
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${conversion_libs}

  MLIRAffineDialect
  MLIRAnalysis
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRExecutionEngine
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRIR
  MLIRLLVMDialect
  MLIRLLVMCommonConversion
  MLIRLLVMToLLVMIRTranslation
  MLIRToLLVMIRTranslationRegistration
  MLIRMemRefDialect
  MLIRMemRefToLLVM
  MLIRNVVMDialect
  MLIRNVVMToLLVMIRTranslation
  MLIRPass
  MLIRROCDLDialect
  MLIRROCDLToLLVMIRTranslation
  MLIRSCFDialect
  MLIRSupport
  MLIRTargetLLVMIRExport
  MLIRTransforms
  MLIRTransformUtils
  MLIRTranslateLib
  MLIRVectorDialect
  MLIRVectorToLLVM
  )

add_mlir_library(MLIRGPUTestPasses
  TestConvertGPUKernelToCubin.cpp
  TestConvertGPUKernelToHsaco.cpp
  TestGpuMemoryPromotion.cpp
  TestGpuRewrite.cpp
  TestLowerToNVVM.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LIBS}
  )
