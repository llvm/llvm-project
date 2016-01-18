add_llvm_library(LLVMHexagonDesc
  HexagonAsmBackend.cpp
  HexagonELFObjectWriter.cpp
  HexagonInstPrinter.cpp
  HexagonMCAsmInfo.cpp
  HexagonMCCodeEmitter.cpp
  HexagonMCCompound.cpp
  HexagonMCDuplexInfo.cpp
  HexagonMCELFStreamer.cpp
  HexagonMCInstrInfo.cpp
  HexagonMCShuffler.cpp
  HexagonMCTargetDesc.cpp
  HexagonShuffler.cpp
  )

add_dependencies(LLVMHexagonDesc HexagonCommonTableGen)
