set(LLVM_TARGET_DEFINITIONS GPUToNVVM.td)
mlir_tablegen(GPUToNVVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToNVVMIncGen)

add_mlir_conversion_library(MLIRGPUtoNVVMTransforms
  LowerGpuOpsToNVVMOps.cpp

  LINK_LIBS
  LLVMCore
  LLVMSupport
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRStandardOps
  MLIRStandardToLLVM
  MLIRSupport
  MLIRTransforms
  MLIRTransformUtils
  )

add_dependencies(MLIRGPUtoNVVMTransforms
  MLIRGPUToNVVMIncGen)
