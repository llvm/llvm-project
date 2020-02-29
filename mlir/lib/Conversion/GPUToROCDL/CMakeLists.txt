add_mlir_conversion_library(MLIRGPUtoROCDLTransforms
  LowerGpuOpsToROCDLOps.cpp

  LINK_LIBS
  LLVMCore
  LLVMSupport
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRROCDLIR
  MLIRPass
  MLIRStandardToLLVM
  MLIRStandardOps
  MLIRTransforms
  MLIRTransformUtils
  )
