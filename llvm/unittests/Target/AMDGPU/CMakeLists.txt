include_directories(
  ${PROJECT_SOURCE_DIR}/lib/Target/AMDGPU
  ${PROJECT_BINARY_DIR}/lib/Target/AMDGPU
  )

set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  AMDGPUUtils
  CodeGen
  CodeGenTypes
  Core
  MC
  Support
  TargetParser
  )

add_llvm_target_unittest(AMDGPUTests
  AMDGPUUnitTests.cpp
  DwarfRegMappings.cpp
  ExecMayBeModifiedBeforeAnyUse.cpp
  )
