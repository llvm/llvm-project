add_llvm_component_library(LLVMConnexDesc
  ConnexMCTargetDesc.cpp
  ConnexAsmBackend.cpp
  ConnexInstPrinter.cpp
  ConnexMCCodeEmitter.cpp
  ConnexELFObjectWriter.cpp

  LINK_COMPONENTS
  ConnexInfo
  MC
  Support

  ADD_TO_COMPONENT
  Connex
)
