add_llvm_component_library(LLVMLoongArchDesc
  LoongArchAsmBackend.cpp
  LoongArchBaseInfo.cpp
  LoongArchELFObjectWriter.cpp
  LoongArchELFStreamer.cpp
  LoongArchInstPrinter.cpp
  LoongArchMCAsmInfo.cpp
  LoongArchMCTargetDesc.cpp
  LoongArchMCCodeEmitter.cpp
  LoongArchMatInt.cpp
  LoongArchTargetStreamer.cpp

  LINK_COMPONENTS
  MC
  LoongArchInfo
  Support

  ADD_TO_COMPONENT
  LoongArch
  )
