set(LLVM_LINK_COMPONENTS
  DebugInfoCodeView
  )

add_llvm_unittest(DebugInfoCodeViewTests
  FieldListVisitorTest.cpp
  GUIDFormatTest.cpp
  RandomAccessVisitorTest.cpp
  TypeHashingTest.cpp
  TypeIndexDiscoveryTest.cpp
  TypeRecordMappingTest.cpp
  )

target_link_libraries(DebugInfoCodeViewTests PRIVATE LLVMTestingSupport)
