add_llvm_component_library(LLVMInArchDesc
  InArchMCTargetDesc.cpp
  InArchMCAsmInfo.cpp
  InArchInstPrinter.cpp
  InArchMCCodeEmitter.cpp
  InArchAsmBackend.cpp
  InArchELFObjectWriter.cpp

  LINK_COMPONENTS
  MC
  Support
  InArchInfo

  ADD_TO_COMPONENT
  InArch
 )