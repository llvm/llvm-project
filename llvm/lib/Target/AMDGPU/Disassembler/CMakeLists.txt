include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMAMDGPUDisassembler
  AMDGPUDisassembler.cpp
  CodeObject.cpp

  LINK_COMPONENTS
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  CodeGenTypes
  MC
  MCDisassembler
  Object
  Support

  ADD_TO_COMPONENT
  AMDGPU
  )
