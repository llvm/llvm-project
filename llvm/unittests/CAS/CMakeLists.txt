set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  ActionCacheTest.cpp
  CASTestConfig.cpp
  ObjectStoreTest.cpp
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
