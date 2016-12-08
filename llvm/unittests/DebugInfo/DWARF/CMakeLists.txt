set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  CodeGen
  Core
  MC
  MIRParser
  Support
  Target
  )

set(DebugInfoSources
  DWARFDebugInfoTest.cpp
  DWARFFormValueTest.cpp
  )

add_llvm_unittest(DebugInfoDWARFTests
  ${DebugInfoSources}
  )
