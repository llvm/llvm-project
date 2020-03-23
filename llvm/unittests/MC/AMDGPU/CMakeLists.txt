set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  MC
  Support
  )

add_llvm_unittest(AMDGPUDwarfTests
  DwarfRegMappings.cpp
  )
