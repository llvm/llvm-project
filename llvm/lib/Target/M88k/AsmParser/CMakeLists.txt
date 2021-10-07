add_llvm_component_library(LLVMM88kAsmParser
  M88kAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Support
  M88kDesc
  M88kInfo

  ADD_TO_COMPONENT
  M88k
  )
