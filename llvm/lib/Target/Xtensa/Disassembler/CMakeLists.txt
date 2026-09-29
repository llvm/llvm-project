add_llvm_component_library(LLVMXtensaDisassembler
  XtensaDisassembler.cpp

  LINK_COMPONENTS
  MCDisassembler
  Support
  XtensaDesc
  XtensaInfo

  ADD_TO_COMPONENT
  Xtensa
  )
