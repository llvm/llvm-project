add_llvm_component_library(LLVMConnexInfo
  ConnexTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  Connex
  )
