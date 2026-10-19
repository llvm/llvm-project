add_llvm_component_library(LLVMAGXInfo
  AGXTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  AGX
  )
