add_llvm_component_library(LLVMNext32Desc
  Next32MCTargetDesc.cpp
  Next32AsmBackend.cpp
  Next32ELFStreamer.cpp
  Next32MCCodeEmitter.cpp
  Next32ELFObjectWriter.cpp
  Next32MCExpr.cpp
  Next32InstPrinter.cpp

  LINK_COMPONENTS
  MC
  MCDisassembler
  Support
  Next32Info
  BinaryFormat

  ADD_TO_COMPONENT
  Next32
  )
