set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  DebugInfoLogicalView
  MC
  MCDisassembler
  TargetParser
  )

add_llvm_unittest_with_input_files(DebugInfoLogicalViewTests
  CodeViewReaderTest.cpp
  CommandLineOptionsTest.cpp
  CompareElementsTest.cpp
  DWARFReaderTest.cpp
  SelectElementsTest.cpp
  LocationRangesTest.cpp
  LogicalElementsTest.cpp
  StringPoolTest.cpp
  WarningInternalTest.cpp
  )

target_link_libraries(DebugInfoLogicalViewTests PRIVATE LLVMTestingSupport)
