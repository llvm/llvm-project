add_llvm_component_library(LLVMSimInfo
  SimTargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  Sim
  )