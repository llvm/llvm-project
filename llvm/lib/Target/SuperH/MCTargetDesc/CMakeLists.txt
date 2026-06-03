add_llvm_component_library(LLVMSuperHDesc
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
