set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsInfos
  AllTargetsDisassemblers
  DebugInfoLogicalView
  MCDisassembler
  )

add_llvm_unittest_with_input_files(DebugInfoLogicalViewTests
  CommandLineOptionsTest.cpp
  CompareElementsTest.cpp
  ELFReaderTest.cpp
  SelectElementsTest.cpp
  LocationRangesTest.cpp
  LogicalElementsTest.cpp
  StringPoolTest.cpp
  WarningInternalTest.cpp
  )

target_link_libraries(DebugInfoLogicalViewTests PRIVATE LLVMTestingSupport)
