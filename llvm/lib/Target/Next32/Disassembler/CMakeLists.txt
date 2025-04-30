add_llvm_component_library(LLVMNext32Disassembler
  Next32Disassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  Support
  Next32Info

  ADD_TO_COMPONENT
  Next32
)

