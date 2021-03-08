add_llvm_component_library(LLVMM68kDesc
  M68kMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  M68kInfo

  ADD_TO_COMPONENT
  M68k
)
