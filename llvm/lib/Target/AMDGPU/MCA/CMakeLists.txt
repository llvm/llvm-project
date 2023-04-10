add_llvm_component_library(LLVMAMDGPUTargetMCA
  AMDGPUCustomBehaviour.cpp

  LINK_COMPONENTS
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  CodeGen
  MC
  MCA
  MCParser
  Support
  TargetParser

  ADD_TO_COMPONENT
  AMDGPU
  )

add_dependencies(LLVMAMDGPUTargetMCA LLVMAMDGPUUtils)
