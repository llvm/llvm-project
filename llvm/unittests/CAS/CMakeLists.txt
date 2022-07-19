set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  CASDBTest.cpp
  CASFileSystemTest.cpp
  CASOutputBackendTest.cpp
  CachingOnDiskFileSystemTest.cpp
  HashMappedTrieTest.cpp
  HierarchicalTreeBuilderTest.cpp
  OnDiskHashMappedTrieTest.cpp
  ThreadSafeAllocatorTest.cpp
  TreeSchemaTest.cpp
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
