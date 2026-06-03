add_llvm_component_library(LLVMSuperHDesc
  SuperHMCCodeEmitter.cpp
  SuperHMCTargetDesc.cpp
  SuperHMCAsmInfo.cpp
  SuperHInstPrinter.cpp
  SuperHELFObjectWriter.cpp
  SuperHAsmBackend.cpp

  LINK_COMPONENTS
  MC
  SuperHInfo
  Support
  TargetParser

  ADD_TO_COMPONENT
  SuperH
)
