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
  )

add_llvm_target_unittest(AMDGPUTests
  DwarfRegMappings.cpp
  ExecMayBeModifiedBeforeAnyUse.cpp
  )

set_property(TARGET AMDGPUTests PROPERTY FOLDER "Tests/UnitTests/TargetTests")
