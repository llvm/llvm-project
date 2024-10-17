if (LLVM_ENABLE_ONDISK_CAS)
  add_definitions(-DLLVM_ENABLE_ONDISK_CAS=1)
endif()

add_llvm_component_library(LLVMCAS
  ActionCache.cpp
  ActionCaches.cpp
  BuiltinCAS.cpp
  BuiltinUnifiedCASDatabases.cpp
  CASNodeSchema.cpp
  CASRegistry.cpp
  HierarchicalTreeBuilder.cpp
  InMemoryCAS.cpp
  MappedFileRegionBumpPtr.cpp
  ObjectStore.cpp
  OnDiskCAS.cpp
  OnDiskCommon.cpp
  OnDiskGraphDB.cpp
  OnDiskHashMappedTrie.cpp
  OnDiskKeyValueDB.cpp
  PluginCAS.cpp
  TreeEntry.cpp
  TreeSchema.cpp
  UnifiedOnDiskCache.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_COMPONENTS
  Support
)
