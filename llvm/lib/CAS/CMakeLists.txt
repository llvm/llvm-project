add_llvm_component_library(LLVMCAS
  ActionCache.cpp
  ActionCaches.cpp
  BuiltinCAS.cpp
  DatabaseFile.cpp
  InMemoryCAS.cpp
  MappedFileRegionArena.cpp
  ObjectStore.cpp
  OnDiskCommon.cpp
  OnDiskDataAllocator.cpp
  OnDiskTrieRawHashMap.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  Support
)
