set(LLVM_TARGET_DEFINITIONS GPUToNVVM.td)
mlir_tablegen(GPUToNVVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToNVVMIncGen)

add_mlir_conversion_library(MLIRGPUToNVVMTransforms
  LowerGpuOpsToNVVMOps.cpp
  WmmaOpsToNvvm.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRGPUToNVVMIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUDialect
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRNVGPUDialect
  MLIRNVVMDialect
  MLIRPass
  MLIRTransformUtils
  )
