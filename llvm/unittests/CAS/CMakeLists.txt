set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  CASTestConfig.cpp
  ObjectStoreTest.cpp
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
