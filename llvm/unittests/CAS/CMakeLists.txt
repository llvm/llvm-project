if (LLVM_CAS_ENABLE_REMOTE_CACHE)
  add_definitions(-DLLVM_CAS_ENABLE_REMOTE_CACHE=1)
  set(ADDITIONAL_CAS_TEST_DEPS "RemoteCacheServer")
endif()

set(ONDISK_CAS_TEST_SOURCES
  BuiltinObjectHasherTest.cpp
  BuiltinUnifiedCASDatabasesTest.cpp
  CASConfigurationTest.cpp
  OnDiskCASLoggerTest.cpp
  OnDiskGraphDBTest.cpp
  OnDiskDataAllocatorTest.cpp
  OnDiskKeyValueDBTest.cpp
  OnDiskTrieRawHashMapTest.cpp
  ProgramTest.cpp
  UnifiedOnDiskCacheTest.cpp
  )

set(PLUGIN_CAS_TEST_SOURCES
  PluginCASTest.cpp
  )

set(REMOTE_CACHE_TEST_SOURCES
  MockGRPCServer.cpp
  )

set(LLVM_OPTIONAL_SOURCES
  ${ONDISK_CAS_TEST_SOURCES}
  ${PLUGIN_CAS_TEST_SOURCES}
  ${REMOTE_CACHE_TEST_SOURCES}
  )

if (NOT LLVM_ENABLE_ONDISK_CAS)
  unset(ONDISK_CAS_TEST_SOURCES)
endif()

# The plugin tests need the libCASPluginTest shared library, which is not
# always built (it requires the on-disk CAS and PIC).
if (NOT TARGET CASPluginTest)
  unset(PLUGIN_CAS_TEST_SOURCES)
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
  CASFSBuilderTest.cpp
  CASTestConfig.cpp
  CASOutputBackendTest.cpp
  CASProvidingFileSystemTest.cpp
  CachingOnDiskFileSystemTest.cpp
  HierarchicalTreeBuilderTest.cpp
  NamedValuesSchemaTest.cpp
  ObjectStoreTest.cpp
  OnDiskCommonUtils.cpp
  TreeSchemaTest.cpp

  ${ONDISK_CAS_TEST_SOURCES}
  ${PLUGIN_CAS_TEST_SOURCES}
  ${REMOTE_CACHE_TEST_SOURCES}
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)

if (TARGET CASPluginTest)
  add_dependencies(CASTests CASPluginTest)
  # Symlink the plugin library next to the test executable so getCASPluginPath()
  # can locate it with a single parent_path regardless of the generator type.
  add_custom_command(TARGET CASTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:CASPluginTest>
            $<TARGET_FILE_DIR:CASTests>/$<TARGET_FILE_PREFIX:CASPluginTest>CASPluginTest$<TARGET_FILE_SUFFIX:CASPluginTest>
    VERBATIM
  )
endif()
