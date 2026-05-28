set(MCPARSER_SOURCES
  AsmLexer.cpp
  AsmParser.cpp
  ELFAsmParser.cpp
  MCAsmParser.cpp
  MCAsmParserExtension.cpp
  MCTargetAsmParser.cpp
  )

set(MCPARSER_OPT_SOURCES
  COFFAsmParser.cpp
  COFFMasmParser.cpp
  GOFFAsmParser.cpp
  DarwinAsmParser.cpp
  MasmParser.cpp
  WasmAsmParser.cpp
  XCOFFAsmParser.cpp
  )

if(NOT EJIT_BARE_METAL)
  list(APPEND MCPARSER_SOURCES ${MCPARSER_OPT_SOURCES})
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
