add_llvm_component_library(LLVMAGXDisassembler
  AGXDisassembler.cpp

  LINK_COMPONENTS
  AGXDesc
  AGXInfo
  AGXUtils
  MC
  MCDisassembler
  Support

  ADD_TO_COMPONENT
  AGX
  )
