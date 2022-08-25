set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  ActionCacheTest.cpp
  CASFileSystemTest.cpp
  CASTestConfig.cpp
  CASOutputBackendTest.cpp
  CASProvidingFileSystemTest.cpp
  CachingOnDiskFileSystemTest.cpp
  HashMappedTrieTest.cpp
  HierarchicalTreeBuilderTest.cpp
  ObjectStoreTest.cpp
  OnDiskHashMappedTrieTest.cpp
  ThreadSafeAllocatorTest.cpp
  TreeSchemaTest.cpp
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
