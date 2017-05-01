add_llvm_library(LLVMObject
  Archive.cpp
  ArchiveWriter.cpp
  Binary.cpp
  COFFObjectFile.cpp
  Decompressor.cpp
  ELF.cpp
  ELFObjectFile.cpp
  Error.cpp
  IRObjectFile.cpp
  IRSymtab.cpp
  MachOObjectFile.cpp
  MachOUniversal.cpp
  ModuleSymbolTable.cpp
  Object.cpp
  ObjectFile.cpp
  RecordStreamer.cpp
  SymbolicFile.cpp
  SymbolSize.cpp
  WasmObjectFile.cpp

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Object

  DEPENDS
  intrinsics_gen
  )
