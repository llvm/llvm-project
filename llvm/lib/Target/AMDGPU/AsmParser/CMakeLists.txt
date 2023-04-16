add_llvm_component_library(LLVMAMDGPUAsmParser
  AMDGPUAsmParser.cpp

  LINK_COMPONENTS
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  MC
  MCParser
  Support
  TargetParser

  ADD_TO_COMPONENT
  AMDGPU
  )

add_dependencies(LLVMAMDGPUAsmParser LLVMAMDGPUUtils)
