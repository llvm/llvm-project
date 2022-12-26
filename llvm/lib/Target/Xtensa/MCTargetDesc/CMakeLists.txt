add_llvm_component_library(LLVMXtensaDesc
  XtensaAsmBackend.cpp
  XtensaELFObjectWriter.cpp
  XtensaMCAsmInfo.cpp
  XtensaMCCodeEmitter.cpp
  XtensaMCTargetDesc.cpp

  LINK_COMPONENTS
  MC
  Support
  XtensaInfo

  ADD_TO_COMPONENT
  Xtensa
 )
