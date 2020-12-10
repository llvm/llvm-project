add_llvm_component_library(LLVMAMDGPUUtils
  AMDGPUBaseInfo.cpp
  AMDKernelCodeTUtils.cpp
  AMDGPUAsmUtils.cpp
  AMDGPUPALMetadata.cpp

  LINK_COMPONENTS
  Core
  MC
  BinaryFormat
  Support

  ADD_TO_COMPONENT
  AMDGPU
  )
