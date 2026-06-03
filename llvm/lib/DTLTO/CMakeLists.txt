add_llvm_component_library(LLVMDTLTO
  DTLTOInputFiles.cpp
  DTLTO.cpp
  DTLTODistributionDriver.cpp

  LINK_COMPONENTS
  BinaryFormat
  Core
  LTO
  Object
  Support
  )
