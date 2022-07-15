# Modifications Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.
# Notified per clause 4(b) of the license.
include_directories( ${CMAKE_CURRENT_BINARY_DIR}/.. ${CMAKE_CURRENT_SOURCE_DIR}/.. )

add_llvm_component_library(LLVMAMDGPUDisassembler
  AMDGPUDisassembler.cpp
  CodeObject.cpp

  LINK_COMPONENTS
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  MC
  MCDisassembler
  Object
  Support

  ADD_TO_COMPONENT
  AMDGPU
  )

add_dependencies(LLVMAMDGPUDisassembler AMDGPUCommonTableGen LLVMAMDGPUUtils)
