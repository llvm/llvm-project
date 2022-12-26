add_llvm_component_library(LLVMXtensaDesc
  XtensaMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  Support
  XtensaInfo

  ADD_TO_COMPONENT
  Xtensa
 )
