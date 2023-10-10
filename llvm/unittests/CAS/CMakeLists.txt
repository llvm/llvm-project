if (LLVM_ENABLE_ONDISK_CAS)
  add_definitions(-DLLVM_ENABLE_ONDISK_CAS=1)
  set(ADDITIONAL_CAS_TEST_DEPS "CASPluginTest")
endif()

set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  ActionCacheTest.cpp
  CASTestConfig.cpp
  HierarchicalTreeBuilderTest.cpp
  ObjectStoreTest.cpp
  OnDiskCommonUtils.h
  OnDiskGraphDBTest.cpp
  OnDiskHashMappedTrieTest.cpp
  OnDiskKeyValueDBTest.cpp
  PluginCASTest.cpp
  TreeSchemaTest.cpp
  UnifiedOnDiskCacheTest.cpp

  DEPENDS
  ${ADDITIONAL_CAS_TEST_DEPS}
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
