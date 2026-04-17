add_llvm_component_library(LLVMDTLTO
  DTLTO.cpp

  LINK_COMPONENTS
  BinaryFormat
  Core
  LTO
  Object
  Support
  )
