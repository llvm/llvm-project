add_llvm_component_library(LLVMAMDGPUTargetMCA
  AMDGPUCustomBehaviour.cpp

  LINK_COMPONENTS
  MC
  MCParser
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  Support
  MCA

  ADD_TO_COMPONENT
  AMDGPU
  )

add_dependencies(LLVMAMDGPUTargetMCA LLVMAMDGPUUtils)