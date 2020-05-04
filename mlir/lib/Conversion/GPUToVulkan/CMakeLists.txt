add_mlir_conversion_library(MLIRGPUtoVulkanTransforms
  ConvertLaunchFuncToVulkanCalls.cpp
  ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

  DEPENDS
  MLIRConversionPassIncGen
  )

target_link_libraries(MLIRGPUtoVulkanTransforms
  PUBLIC
  MLIRGPU
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRStandardOps
  MLIRSupport
  MLIRTransforms
  MLIRTranslation
  LLVMSupport
  )
