if (LLVM_CAS_ENABLE_REMOTE_CACHE)
  add_definitions(-DLLVM_CAS_ENABLE_REMOTE_CACHE=1)
  set(ADDITIONAL_CAS_TEST_DEPS "RemoteCacheServer")
endif()

set(ONDISK_CAS_TEST_SOURCES
  BuiltinUnifiedCASDatabasesTest.cpp
  CASConfigurationTest.cpp
  OnDiskCASLoggerTest.cpp
  OnDiskGraphDBTest.cpp
  OnDiskDataAllocatorTest.cpp
  OnDiskKeyValueDBTest.cpp
  OnDiskTrieRawHashMapTest.cpp
  PluginCASTest.cpp
  ProgramTest.cpp
  UnifiedOnDiskCacheTest.cpp
  )

set(REMOTE_CACHE_TEST_SOURCES
  MockGRPCServer.cpp
  )

set(LLVM_OPTIONAL_SOURCES
  ${ONDISK_CAS_TEST_SOURCES}
  ${REMOTE_CACHE_TEST_SOURCES}
  )

if (NOT LLVM_ENABLE_ONDISK_CAS)
  unset(ONDISK_CAS_TEST_SOURCES)
endif()

if (NOT LLVM_CAS_ENABLE_REMOTE_CACHE)
  unset(REMOTE_CACHE_TEST_SOURCES)
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
  CASFileSystemTest.cpp
  CASTestConfig.cpp
  CASOutputBackendTest.cpp
  CASProvidingFileSystemTest.cpp
  CachingOnDiskFileSystemTest.cpp
  HierarchicalTreeBuilderTest.cpp
  NamedValuesSchemaTest.cpp
  ObjectStoreTest.cpp
  ThreadSafeAllocatorTest.cpp
  TreeSchemaTest.cpp

  ${ONDISK_CAS_TEST_SOURCES}
  ${REMOTE_CACHE_TEST_SOURCES}
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
add_dependencies(CASTests CASPluginTest)
