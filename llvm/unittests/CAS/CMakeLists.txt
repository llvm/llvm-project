set(ONDISK_CAS_TEST_SOURCES
  BuiltinObjectHasherTest.cpp
  BuiltinUnifiedCASDatabasesTest.cpp
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

set(LLVM_OPTIONAL_SOURCES
  ${ONDISK_CAS_TEST_SOURCES}
  ${PLUGIN_CAS_TEST_SOURCES}
  )

if (NOT LLVM_ENABLE_ONDISK_CAS)
  unset(ONDISK_CAS_TEST_SOURCES)
endif()

add_subdirectory(CASPluginTest)

if (NOT TARGET CASPluginTest)
  unset(PLUGIN_CAS_TEST_SOURCES)
endif()

set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  ActionCacheTest.cpp
  CASTestConfig.cpp
  NamedValuesSchemaTest.cpp
  ObjectStoreTest.cpp
  OnDiskCommonUtils.cpp

  ${ONDISK_CAS_TEST_SOURCES}
  ${PLUGIN_CAS_TEST_SOURCES}
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)

if (TARGET CASPluginTest)
  add_dependencies(CASTests CASPluginTest)
  # Symlink the plugin library next to the test executable so getCASPluginPath()
  # can locate it with a single parent_path regardless of the generator type.
  add_custom_command(TARGET CASTests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            $<TARGET_FILE:CASPluginTest>
            $<TARGET_FILE_DIR:CASTests>/$<TARGET_FILE_NAME:CASPluginTest>
    VERBATIM
  )
endif()
