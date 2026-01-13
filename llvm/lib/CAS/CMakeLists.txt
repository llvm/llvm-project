if (UNIX AND "${CMAKE_SYSTEM_NAME}" MATCHES "AIX")
  set(additional_libs bsd)
endif()

add_llvm_component_library(LLVMCAS
  ActionCache.cpp
  ActionCaches.cpp
  BuiltinCAS.cpp
  BuiltinUnifiedCASDatabases.cpp
  CASConfiguration.cpp
  CASFileSystem.cpp
  CASNodeSchema.cpp
  CASOutputBackend.cpp
  CASProvidingFileSystem.cpp
  CachingOnDiskFileSystem.cpp
  DatabaseFile.cpp
  FileSystemCache.cpp
  HierarchicalTreeBuilder.cpp
  InMemoryCAS.cpp
  MappedFileRegionArena.cpp
  NamedValuesSchema.cpp
  ObjectStore.cpp
  OnDiskCAS.cpp
  OnDiskCASLogger.cpp
  OnDiskCommon.cpp
  OnDiskDataAllocator.cpp
  OnDiskGraphDB.cpp
  OnDiskTrieRawHashMap.cpp
  OnDiskKeyValueDB.cpp
  PluginCAS.cpp
  TreeSchema.cpp
  UnifiedOnDiskCache.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}
  ${additional_libs}

  LINK_COMPONENTS
  Support
)
