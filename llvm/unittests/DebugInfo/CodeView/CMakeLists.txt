set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  )

add_llvm_unittest(DebugInfoCodeViewTests
  GUIDFormatTest.cpp
  RandomAccessVisitorTest.cpp
  TypeHashingTest.cpp
  TypeIndexDiscoveryTest.cpp
  )

target_link_libraries(DebugInfoCodeViewTests PRIVATE LLVMTestingSupport)
