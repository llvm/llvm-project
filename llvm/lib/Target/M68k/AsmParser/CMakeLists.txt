add_llvm_component_library(LLVMM68kAsmParser
  M68kAsmParser.cpp

  LINK_COMPONENTS
  MC
  MCParser
  Support
  M68kCodeGen
  M68kInfo

  ADD_TO_COMPONENT
  M68k
)
