add_mlir_conversion_library(MLIRGPUToVulkanTransforms
  ConvertLaunchFuncToVulkanCalls.cpp
  ConvertGPULaunchFuncToVulkanLaunchFunc.cpp

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRFuncDialect
  MLIRGPUOps
  MLIRIR
  MLIRLLVMDialect
  MLIRPass
  MLIRSPIRVDialect
  MLIRSPIRVSerialization
  MLIRSupport
  MLIRTransforms
  MLIRTranslateLib
  )
