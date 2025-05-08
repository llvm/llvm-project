# Modified by Sunscreen under the AGPLv3 license; see the README at the
# repository root for more information

add_llvm_component_library(LLVMParasolDesc
  ParasolAsmBackend.cpp
  ParasolELFObjectWriter.cpp
  ParasolMCAsmInfo.cpp
  ParasolMCTargetDesc.cpp
  ParasolMCCodeEmitter.cpp
  ParasolInstPrinter.cpp

  LINK_COMPONENTS
  MC
  ParasolInfo
  Support
  TargetParser


  ADD_TO_COMPONENT
  Parasol
)
