add_llvm_component_library(LLVMAMDGPUUtils
  AMDGPUAsmUtils.cpp
  AMDGPUBaseInfo.cpp
  AMDGPUMemoryUtils.cpp
  AMDGPUPALMetadata.cpp
  AMDKernelCodeTUtils.cpp

  LINK_COMPONENTS
  Analysis
  BinaryFormat
  Core
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  AMDGPU
  )
