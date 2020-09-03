set(LLVM_LINK_COMPONENTS
  AMDGPUCodeGen
  AMDGPUDesc
  AMDGPUInfo
  MC
  Support
  )

add_llvm_unittest(AMDGPUDwarfTests
  DwarfRegMappings.cpp
  )
