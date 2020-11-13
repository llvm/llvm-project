add_llvm_component_library(LLVMNVPTXInfo
  NVPTXTargetInfo.cpp

  LINK_COMPONENTS
  Support

  ADD_TO_COMPONENT
  NVPTX
  )
