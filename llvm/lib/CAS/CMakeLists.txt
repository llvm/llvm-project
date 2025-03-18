if (LLVM_ENABLE_ONDISK_CAS)
  add_definitions(-DLLVM_ENABLE_ONDISK_CAS=1)
endif()

add_llvm_component_library(LLVMCAS
  ActionCache.cpp
  ActionCaches.cpp
  BuiltinCAS.cpp
  BuiltinUnifiedCASDatabases.cpp
  CASFileSystem.cpp
  CASNodeSchema.cpp
  CASOutputBackend.cpp
  CASProvidingFileSystem.cpp
  CachingOnDiskFileSystem.cpp
  FileSystemCache.cpp
  HashMappedTrie.cpp
  HierarchicalTreeBuilder.cpp
  InMemoryCAS.cpp
  MappedFileRegionBumpPtr.cpp
  ObjectStore.cpp
  OnDiskCAS.cpp
  OnDiskCASLogger.cpp
  OnDiskCommon.cpp
  OnDiskGraphDB.cpp
  OnDiskHashMappedTrie.cpp
  OnDiskKeyValueDB.cpp
  PluginCAS.cpp
  TreeSchema.cpp
  UnifiedOnDiskCache.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_COMPONENTS
  BinaryFormat
  Support
)
