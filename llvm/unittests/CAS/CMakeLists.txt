if (LLVM_ENABLE_ONDISK_CAS)
  add_definitions(-DLLVM_ENABLE_ONDISK_CAS=1)
endif()

if (LLVM_CAS_ENABLE_REMOTE_CACHE)
  add_definitions(-DLLVM_CAS_ENABLE_REMOTE_CACHE=1)
  set(ADDITIONAL_CAS_TEST_DEPS "RemoteCacheServer")
endif()

set(LLVM_LINK_COMPONENTS
  Support
  CAS
  RemoteCachingService
  TestingSupport
  ${ADDITIONAL_CAS_TEST_DEPS}
  )

add_llvm_unittest(CASTests
  ActionCacheTest.cpp
  BuiltinUnifiedCASDatabasesTest.cpp
  CASFileSystemTest.cpp
  CASTestConfig.cpp
  CASOutputBackendTest.cpp
  CASProvidingFileSystemTest.cpp
  CachingOnDiskFileSystemTest.cpp
  HashMappedTrieTest.cpp
  HierarchicalTreeBuilderTest.cpp
  MockGRPCServer.cpp
  ObjectStoreTest.cpp
  OnDiskCASLoggerTest.cpp
  OnDiskGraphDBTest.cpp
  OnDiskHashMappedTrieTest.cpp
  OnDiskKeyValueDBTest.cpp
  PluginCASTest.cpp
  ThreadSafeAllocatorTest.cpp
  TreeSchemaTest.cpp
  UnifiedOnDiskCacheTest.cpp
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
add_dependencies(CASTests CASPluginTest)
