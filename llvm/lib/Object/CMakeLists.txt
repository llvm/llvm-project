set(LLVMOBJECT_SOURCES
  Archive.cpp
  ArchiveWriter.cpp
  Binary.cpp
  BuildID.cpp
  Decompressor.cpp
  ELF.cpp
  ELFObjectFile.cpp
  Error.cpp
  FaultMapParser.cpp
  IRObjectFile.cpp
  IRSymtab.cpp
  ModuleSymbolTable.cpp
  Object.cpp
  ObjectFile.cpp
  RecordStreamer.cpp
  RelocationResolver.cpp
  SymbolicFile.cpp
  SymbolSize.cpp
  WindowsMachineFlag.cpp
  )

# OS/format-specific files not needed for bare-metal EJIT
set(LLVMOBJECT_OPT_SOURCES
  COFFImportFile.cpp
  COFFModuleDefinition.cpp
  COFFObjectFile.cpp
  DXContainer.cpp
  GOFFObjectFile.cpp
  MachOObjectFile.cpp
  MachOUniversal.cpp
  MachOUniversalWriter.cpp
  Minidump.cpp
  OffloadBinary.cpp
  OffloadBundle.cpp
  TapiFile.cpp
  TapiUniversal.cpp
  WasmObjectFile.cpp
  WindowsResource.cpp
  XCOFFObjectFile.cpp
  )

set(LLVMOBJECT_LINK
  BitReader
  Core
  MC
  IRReader
  BinaryFormat
  MCParser
  Support
  TargetParser
  )

if(NOT EJIT_BARE_METAL)
  list(APPEND LLVMOBJECT_SOURCES ${LLVMOBJECT_OPT_SOURCES})
  list(APPEND LLVMOBJECT_LINK TextAPI)
endif()

add_llvm_component_library(LLVMObject
  ${LLVMOBJECT_SOURCES}

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/Object

  DEPENDS
  intrinsics_gen
  llvm_vcsrevision_h

  LINK_COMPONENTS
  ${LLVMOBJECT_LINK}
  )
