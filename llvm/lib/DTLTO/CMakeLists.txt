add_llvm_component_library(LLVMDTLTO
  DTLTOInputFiles.cpp
  DTLTO.cpp
  DTLTODistributionDriver.cpp

  LINK_LIBS
  ${LLVM_PTHREAD_LIB}

  LINK_COMPONENTS
  BinaryFormat
  Core
  LTO
  Object
  Support
  )
