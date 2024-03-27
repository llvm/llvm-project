add_llvm_component_library(LLVMInArchInfo
  InArchTargetInfo.cpp

  LINK_COMPONENTS
  MC
  Support

  ADD_TO_COMPONENT
  InArch
  )