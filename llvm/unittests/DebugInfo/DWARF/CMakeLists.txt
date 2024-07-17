set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  BinaryFormat
  CodeGenTypes
  DebugInfoDWARF
  MC
  Object
  ObjectYAML
  Support
  TargetParser
  )

add_llvm_unittest(DebugInfoDWARFTests
  DwarfGenerator.cpp
  DwarfUtils.cpp
  DWARFAcceleratorTableTest.cpp
  DWARFDataExtractorTest.cpp
  DWARFDebugAbbrevTest.cpp
  DWARFDebugArangeSetTest.cpp
  DWARFDebugFrameTest.cpp
  DWARFDebugInfoTest.cpp
  DWARFDebugLineTest.cpp
  DWARFDieTest.cpp
  DWARFDieManualExtractTest.cpp
  DWARFExpressionCopyBytesTest.cpp
  DWARFExpressionCompactPrinterTest.cpp
  DWARFFormValueTest.cpp
  DWARFListTableTest.cpp
  DWARFLocationExpressionTest.cpp
  )

target_link_libraries(DebugInfoDWARFTests PRIVATE LLVMTestingSupport)
