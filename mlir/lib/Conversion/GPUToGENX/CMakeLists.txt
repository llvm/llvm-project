set(LLVM_TARGET_DEFINITIONS GPUToGENX.td)
mlir_tablegen(GPUToGENX.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToGENXIncGen)

add_mlir_conversion_library(MLIRGPUToGENXTransforms
  LowerGpuOpsToGENXOps.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRGPUToGENXIncGen

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRFuncToLLVM
  MLIRGPUOps
  MLIRGPUToGPURuntimeTransforms
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRMemRefToLLVM
  MLIRGENXDialect
  MLIRPass
  )
