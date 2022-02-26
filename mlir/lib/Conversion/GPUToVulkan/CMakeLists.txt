add_mlir_conversion_library(MLIRGPUToVulkanTransforms
  ConvertLaunchFuncToVulkanCalls.cpp
  ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFunc
  MLIRGPUOps
  MLIRIR
  MLIRLLVMIR
  MLIRPass
  MLIRSPIRV
  MLIRSPIRVSerialization
  MLIRSupport
  MLIRTransforms
  MLIRTranslation
  )
