add_llvm_component_library(LLVMCAS
  BuiltinCAS.cpp
  InMemoryCAS.cpp
  MappedFileRegionBumpPtr.cpp
  ObjectStore.cpp
  OnDiskCommon.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_COMPONENTS
  Support
)
