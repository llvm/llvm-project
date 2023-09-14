# Exclude tests from libMLIR.so
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS
  ${conversion_libs}

  MLIRAffineDialect
  MLIRArithDialect
  MLIRBuiltinToLLVMIRTranslation
  MLIRFuncDialect
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRGPUTransforms
  MLIRIR
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
  MLIRVectorToLLVMPass
  )

add_mlir_library(MLIRGPUTestPasses
  TestGpuMemoryPromotion.cpp
  TestGpuRewrite.cpp
  TestLowerToNVVM.cpp

  EXCLUDE_FROM_LIBMLIR

  LINK_LIBS PUBLIC
  ${LIBS}
  )

