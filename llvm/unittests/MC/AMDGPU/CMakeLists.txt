set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  MC
  Support
  TargetParser
  )

add_llvm_unittest(AMDGPUDwarfTests
  DwarfRegMappings.cpp
  )
