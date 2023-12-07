add_mlir_dialect_library(MLIRGPUTransformOps
  GPUTransformOps.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/GPU/TransformOps
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Interfaces

  DEPENDS
  MLIRGPUTransformOpsIncGen
  MLIRDeviceMappingInterfacesIncGen
  MLIRGPUDeviceMapperEnumsGen
  
  LINK_LIBS PUBLIC
  MLIRGPUDialect
  MLIRGPUTransforms
  MLIRIR
  MLIRParser
  MLIRSideEffectInterfaces
  MLIRTransformDialect
  MLIRVectorDialect
  MLIRVectorTransforms

  # ConversionPatterns
  MLIRNVGPUToNVVM
  MLIRGPUToNVVMTransforms
  )  
