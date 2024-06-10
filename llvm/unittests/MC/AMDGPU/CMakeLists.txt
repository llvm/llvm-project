include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU
  ${PROJECT_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  CodeGen
  Core
  MC
  Support
  TargetParser
  )

add_llvm_unittest(AMDGPUMCTests
  DwarfRegMappings.cpp
  SIProgramInfoMCExprs.cpp
  )
