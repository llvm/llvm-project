add_llvm_component_library(LLVMLanaiInfo
  LanaiTargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  Lanai
  )
