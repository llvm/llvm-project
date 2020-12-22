add_llvm_component_library(LLVMCSKYDesc
  CSKYAsmBackend.cpp
  CSKYELFObjectWriter.cpp
  CSKYMCAsmInfo.cpp
  CSKYMCTargetDesc.cpp
  CSKYMCCodeEmitter.cpp

  LINK_COMPONENTS
  MC
  CSKYInfo
  Support

  ADD_TO_COMPONENT
  CSKY
)
