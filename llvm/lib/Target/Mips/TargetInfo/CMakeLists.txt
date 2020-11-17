add_llvm_component_library(LLVMMipsInfo
  MipsTargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  Mips
  )
