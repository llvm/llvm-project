add_llvm_component_library(LLVMAGXDesc
  AGXAsmBackend.cpp
  AGXELFObjectWriter.cpp
  AGXInstPrinter.cpp
  AGXMCAsmInfo.cpp
  AGXMCCodeEmitter.cpp
  AGXMCTargetDesc.cpp
  AGXMesaISA.cpp

  LINK_COMPONENTS
  AGXInfo
  AGXUtils
  CodeGenTypes
  MC
  Support
  TargetParser

  ADD_TO_COMPONENT
  AGX
  )

add_dependencies(LLVMAGXDesc AGXCommonTableGen)
