add_llvm_component_library(LLVMARMDesc
  ARMAsmBackend.cpp
  ARMELFObjectWriter.cpp
  ARMELFStreamer.cpp
  ARMInstPrinter.cpp
  ARMMachObjectWriter.cpp
  ARMMachORelocationInfo.cpp
  ARMMCAsmInfo.cpp
  ARMMCCodeEmitter.cpp
  ARMMCTargetDesc.cpp
  ARMTargetStreamer.cpp
  ARMUnwindOpAsm.cpp
  ARMWinCOFFObjectWriter.cpp
  ARMWinCOFFStreamer.cpp

  LINK_COMPONENTS
  ARMInfo
  ARMUtils
  BinaryFormat
  CodeGenTypes
  MC
  MCDisassembler
  Object
  Support
  TargetParser

  ADD_TO_COMPONENT
  ARM
  )

# Static ELF relocation helper functions conflict in unity builds.
set_source_files_properties(
  ARMELFObjectWriter.cpp
  PROPERTIES SKIP_UNITY_BUILD_INCLUSION ON)
