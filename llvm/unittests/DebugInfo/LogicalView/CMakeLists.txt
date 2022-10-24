set(LLVM_LINK_COMPONENTS
  DebugInfoLogicalView
  )

add_llvm_unittest(DebugInfoLogicalViewTests
  CommandLineOptionsTest.cpp
  CompareElementsTest.cpp
  SelectElementsTest.cpp
  LocationRangesTest.cpp
  LogicalElementsTest.cpp
  StringPoolTest.cpp
  WarningInternalTest.cpp
  )

target_link_libraries(DebugInfoLogicalViewTests PRIVATE LLVMTestingSupport)
