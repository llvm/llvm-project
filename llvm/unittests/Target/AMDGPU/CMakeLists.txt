include_directories(
  ${CMAKE_SOURCE_DIR}/lib/Target/AMDGPU
  ${CMAKE_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  MC
  Support
  )

add_llvm_target_unittest(AMDGPUTests
  DwarfRegMappings.cpp
  )
