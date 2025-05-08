# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

add_llvm_component_library(LLVMParasolDisassembler
  ParasolDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  MC
  ParasolInfo
  Support

  ADD_TO_COMPONENT
  Parasol
  )
