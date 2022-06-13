add_mlir_conversion_library(MLIRVectorToGPU
  VectorToGPU.cpp
  NvGpuSupport.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/VectorToGPU

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRArithmeticDialect
  MLIRGPUOps
  MLIRLLVMDialect
  MLIRMemRefDialect
  MLIRNVGPUDialect
  MLIRTransforms
  MLIRVectorDialect
  MLIRVectorUtils
  )
