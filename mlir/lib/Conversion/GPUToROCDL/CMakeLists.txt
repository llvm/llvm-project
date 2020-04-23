set(LLVM_TARGET_DEFINITIONS GPUToROCDL.td)
mlir_tablegen(GPUToROCDL.cpp.inc -gen-rewriters)
add_public_tablegen_target(MLIRGPUToROCDLIncGen)

add_mlir_conversion_library(MLIRGPUtoROCDLTransforms
  LowerGpuOpsToROCDLOps.cpp

  DEPENDS
  MLIRConversionPassIncGen
  MLIRGPUToROCDLIncGen
  )

target_link_libraries(MLIRGPUtoROCDLTransforms
  PUBLIC
  MLIRGPU
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRPass
  MLIRStandardToLLVM
  )
