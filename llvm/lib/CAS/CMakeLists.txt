if (LLVM_ENABLE_ONDISK_CAS)
  add_definitions(-DLLVM_ENABLE_ONDISK_CAS=1)
endif()

add_llvm_component_library(LLVMCAS
  ActionCache.cpp
  ActionCaches.cpp
  BuiltinCAS.cpp
  CASFileSystem.cpp
  CASNodeSchema.cpp
  CASOutputBackend.cpp
  CASProvidingFileSystem.cpp
  CachingOnDiskFileSystem.cpp
  FileSystemCache.cpp
  HashMappedTrie.cpp
  HierarchicalTreeBuilder.cpp
  InMemoryCAS.cpp
  LazyMappedFileRegion.cpp
  LazyMappedFileRegionBumpPtr.cpp
  ObjectStore.cpp
  OnDiskCAS.cpp
  OnDiskHashMappedTrie.cpp
  TreeSchema.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_COMPONENTS
  BinaryFormat
)
