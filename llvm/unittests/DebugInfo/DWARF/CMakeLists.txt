set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  DebugInfoDWARF
  MC
  Object
  Support
  )

set(DebugInfoSources
  DwarfGenerator.cpp
  DWARFDebugInfoTest.cpp
  DWARFFormValueTest.cpp
  )

add_llvm_unittest(DebugInfoDWARFTests
  ${DebugInfoSources}
  )
