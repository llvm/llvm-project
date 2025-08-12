add_llvm_component_library(LLVMCAS
  BuiltinCAS.cpp
  InMemoryCAS.cpp
  ObjectStore.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/CAS

  LINK_COMPONENTS
  Support
)
