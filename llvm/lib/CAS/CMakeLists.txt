add_llvm_component_library(LLVMCAS
  BuiltinCAS.cpp
  CASDB.cpp
  CASFileSystem.cpp
  CASNodeSchema.cpp
  CASOutputBackend.cpp
  CachingOnDiskFileSystem.cpp
  FileSystemCache.cpp
  HashMappedTrie.cpp
  HierarchicalTreeBuilder.cpp
  InMemoryCAS.cpp
  LazyMappedFileRegion.cpp
  LazyMappedFileRegionBumpPtr.cpp
  OnDiskCAS.cpp
  OnDiskHashMappedTrie.cpp
  TreeSchema.cpp
  Utils.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_COMPONENTS
  BinaryFormat
)
