add_llvm_component_library(LLVMSuperHInfo
  SuperHTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  SuperH
  )
