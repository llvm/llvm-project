if (LLVM_ENABLE_ONDISK_CAS)
  add_definitions(-DLLVM_ENABLE_ONDISK_CAS=1)
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
  TreeSchemaTest.cpp
  UnifiedOnDiskCacheTest.cpp
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
