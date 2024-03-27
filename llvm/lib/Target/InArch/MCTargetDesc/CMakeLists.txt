add_llvm_component_library(LLVMInArchDesc
  InArchMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  Support
  InArchInfo

  ADD_TO_COMPONENT
  InArch
 )