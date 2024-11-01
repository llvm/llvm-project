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

# This is how it is defined in mlir/lib/Dialect/GPU/CMakeLists.txt
# We probably want something better project-wide
if(MLIR_ENABLE_CUDA_RUNNER)
  # Enable gpu-to-cubin pass.
  target_compile_definitions(MLIRGPUTestPasses
    PRIVATE
    MLIR_GPU_TO_CUBIN_PASS_ENABLE=1
  )
endif()
