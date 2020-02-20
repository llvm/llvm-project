set(LLVM_TARGET_DEFINITIONS GPUToNVVM.td)
mlir_tablegen(GPUToNVVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToNVVMIncGen)

add_mlir_conversion_library(MLIRGPUtoNVVMTransforms
  LowerGpuOpsToNVVMOps.cpp

  DEPENDS
  MLIRGPUToNVVMIncGen
  )

target_link_libraries(MLIRGPUtoNVVMTransforms
  PUBLIC
  LLVMSupport
  MLIRGPU
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransformUtils
  )
