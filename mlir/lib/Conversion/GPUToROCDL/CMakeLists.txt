add_mlir_conversion_library(MLIRGPUtoROCDLTransforms
  LowerGpuOpsToROCDLOps.cpp

  DEPENDS
  MLIRConversionPassIncGen
  )
target_link_libraries(MLIRGPUtoROCDLTransforms
  PUBLIC
  LLVMSupport
  MLIRGPU
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRPass
  MLIRStandardToLLVM
  )
