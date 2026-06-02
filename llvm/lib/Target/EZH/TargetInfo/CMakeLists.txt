add_llvm_component_library(LLVMEZHInfo
  EZHTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  EZH
  )
