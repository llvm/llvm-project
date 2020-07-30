set(LLVM_TARGET_DEFINITIONS GPUToNVVM.td)
mlir_tablegen(GPUToNVVM.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToNVVMIncGen)

add_mlir_conversion_library(MLIRGPUToNVVMTransforms
  LowerGpuOpsToNVVMOps.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRGPUToNVVMIncGen

  LINK_LIBS PUBLIC
  MLIRGPU
  MLIRLLVMIR
  MLIRNVVMIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRTransformUtils
  )
