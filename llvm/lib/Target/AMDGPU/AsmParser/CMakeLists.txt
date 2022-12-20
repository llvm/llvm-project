add_llvm_component_library(LLVMAMDGPUAsmParser
  AMDGPUAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  Support
  TargetParser

  ADD_TO_COMPONENT
  AMDGPU
  )

add_dependencies(LLVMAMDGPUAsmParser LLVMAMDGPUUtils)
