add_llvm_component_library(LLVMVEInfo
  VETargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  VE
  )
