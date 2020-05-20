add_mlir_conversion_library(MLIRGPUtoVulkanTransforms
  ConvertLaunchFuncToVulkanCalls.cpp
  ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
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
  )
