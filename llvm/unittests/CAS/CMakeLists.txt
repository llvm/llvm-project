set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  CASDBTest.cpp
  CASFileSystemTest.cpp
  CASOutputBackendTest.cpp
  CASReferenceTest.cpp
  CASUtilsTest.cpp
  CachingOnDiskFileSystemTest.cpp
  HashMappedTrieTest.cpp
  HierarchicalTreeBuilderTest.cpp
  OnDiskHashMappedTrieTest.cpp
  ThreadSafeAllocatorTest.cpp
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
