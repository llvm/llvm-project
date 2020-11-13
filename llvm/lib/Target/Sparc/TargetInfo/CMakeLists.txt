add_llvm_component_library(LLVMSparcInfo
  SparcTargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  Sparc
  )
