add_llvm_component_library(LLVMXtensaUtils
  XtensaBaseInfo.cpp

  LINK_COMPONENTS
  Support
  Core

  ADD_TO_COMPONENT
  Xtensa
  )
