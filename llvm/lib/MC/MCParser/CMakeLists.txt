set(MCPARSER_SOURCES
  AsmLexer.cpp
  AsmParser.cpp
  ELFAsmParser.cpp
  MCAsmParser.cpp
  MCAsmParserExtension.cpp
  MCTargetAsmParser.cpp
  )

# MachO/COFF parsers — needed by x86
set(MCPARSER_MACHO_COFF
  COFFAsmParser.cpp
  COFFMasmParser.cpp
  DarwinAsmParser.cpp
  MasmParser.cpp
  )

# Bloat — not needed by any EJIT arch
set(MCPARSER_BLOAT
  GOFFAsmParser.cpp
  WasmAsmParser.cpp
  XCOFFAsmParser.cpp
  )

if(NOT EJIT_BARE_METAL)
  list(APPEND MCPARSER_SOURCES ${MCPARSER_MACHO_COFF})
  list(APPEND MCPARSER_SOURCES ${MCPARSER_BLOAT})
elseif(LLVM_TARGETS_TO_BUILD STREQUAL "AArch64")
  # AArch64 bare-metal: only need ELF
else()
  # x86 bare-metal: need ELF + MachO + COFF
  list(APPEND MCPARSER_SOURCES ${MCPARSER_MACHO_COFF})
endif()

add_llvm_component_library(LLVMMCParser
  ${MCPARSER_SOURCES}

  PARTIAL_SOURCES_INTENDED

  ADDITIONAL_HEADER_DIRS
  ${LLVM_MAIN_INCLUDE_DIR}/llvm/MC/MCParser

  LINK_COMPONENTS
  MC
  Support
  TargetParser
  )
