set(ONDISK_CAS_TEST_SOURCES
  BuiltinUnifiedCASDatabasesTest.cpp
  OnDiskGraphDBTest.cpp
  OnDiskDataAllocatorTest.cpp
  OnDiskKeyValueDBTest.cpp
  OnDiskTrieRawHashMapTest.cpp
  ProgramTest.cpp
  UnifiedOnDiskCacheTest.cpp
  )

set(LLVM_OPTIONAL_SOURCES
  ${ONDISK_CAS_TEST_SOURCES}
  )

if (NOT LLVM_ENABLE_ONDISK_CAS)
  unset(ONDISK_CAS_TEST_SOURCES)
endif()

set(LLVM_LINK_COMPONENTS
  Support
  CAS
  TestingSupport
  )

add_llvm_unittest(CASTests
  ActionCacheTest.cpp
  CASTestConfig.cpp
  ObjectStoreTest.cpp

  ${ONDISK_CAS_TEST_SOURCES}
  )

target_link_libraries(CASTests PRIVATE LLVMTestingSupport)
