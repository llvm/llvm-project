set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  DebugInfoDWARF
  MC
  Object
  ObjectYAML
  Support
  )

set(DebugInfoSources
  DwarfGenerator.cpp
  DwarfUtils.cpp
  DWARFDebugInfoTest.cpp
  DWARFDebugLineTest.cpp
  DWARFFormValueTest.cpp
  )

add_llvm_unittest(DebugInfoDWARFTests
  ${DebugInfoSources}
  )

target_link_libraries(DebugInfoDWARFTests PRIVATE LLVMTestingSupport)
