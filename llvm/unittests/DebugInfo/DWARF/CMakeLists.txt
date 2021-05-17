set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  BinaryFormat
  DebugInfoDWARF
  MC
  Object
  ObjectYAML
  Support
  )

add_llvm_unittest(DebugInfoDWARFTests
  DwarfGenerator.cpp
  DwarfUtils.cpp
  DWARFAcceleratorTableTest.cpp
  DWARFDataExtractorTest.cpp
  DWARFDebugArangeSetTest.cpp
  DWARFDebugFrameTest.cpp
  DWARFDebugInfoTest.cpp
  DWARFDebugLineTest.cpp
  DWARFDieTest.cpp
  DWARFDieManualExtractTest.cpp
  DWARFExpressionCompactPrinterTest.cpp
  DWARFFormValueTest.cpp
  DWARFListTableTest.cpp
  DWARFLocationExpressionTest.cpp
  )

target_link_libraries(DebugInfoDWARFTests PRIVATE LLVMTestingSupport)
