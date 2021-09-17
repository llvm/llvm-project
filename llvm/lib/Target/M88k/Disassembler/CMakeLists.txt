add_llvm_component_library(LLVMM88kDisassembler
  M88kDisassembler.cpp

  LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  M88kDesc
  M88kInfo

  ADD_TO_COMPONENT
  M88k
  )
