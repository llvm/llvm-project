# This tool depends on AMDGPU backend internals (GCNSubtarget). Only build it
# when the AMDGPU target is enabled.
if(NOT "AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  return()
endif()

include_directories(
  ${LLVM_MAIN_SRC_DIR}/lib/Target/AMDGPU
  ${LLVM_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  CodeGen
  Core
  MC
  Support
  Target
  TargetParser
  )

add_llvm_tool(llvm-calc-occupancy
  DISABLE_LLVM_LINK_LLVM_DYLIB
  llvm-calc-occupancy.cpp

  DEPENDS
  AMDGPUCommonTableGen
  )
