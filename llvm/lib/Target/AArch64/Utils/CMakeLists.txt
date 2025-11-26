add_llvm_component_library(LLVMAArch64Utils
  AArch64BaseInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  AArch64
  )
