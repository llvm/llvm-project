set(ASMPRINTER_SOURCES
  AIXException.cpp
  ARMException.cpp
  AsmPrinter.cpp
  AsmPrinterDwarf.cpp
  AsmPrinterInlineAsm.cpp
  PseudoProbePrinter.cpp
  EHStreamer.cpp
  ErlangGCPrinter.cpp
  OcamlGCPrinter.cpp
  WasmException.cpp
)

set(ASMPRINTER_DEBUG_SOURCES
  AccelTable.cpp
  AddressPool.cpp
  CodeViewDebug.cpp
  DbgEntityHistoryCalculator.cpp
  DebugHandlerBase.cpp
  DebugLocStream.cpp
  DIE.cpp
  DIEHash.cpp
  DwarfCFIException.cpp
  DwarfCompileUnit.cpp
  DwarfDebug.cpp
  DwarfExpression.cpp
  DwarfFile.cpp
  DwarfStringPool.cpp
  DwarfUnit.cpp
  WinCFGuard.cpp
  WinException.cpp
)

if(EJIT_BARE_METAL)
  message(STATUS "AsmPrinter: EJIT bare-metal mode — excluding DWARF/CodeView/Win")
  set(SOURCES ${ASMPRINTER_SOURCES})
  set(LINK_COMPS Analysis BinaryFormat CodeGen CodeGenTypes Core
                 MC Remarks Support Target TargetParser)
  add_compile_definitions(EJIT_BARE_METAL)
else()
  set(SOURCES ${ASMPRINTER_SOURCES} ${ASMPRINTER_DEBUG_SOURCES})
  set(LINK_COMPS Analysis BinaryFormat CodeGen CodeGenTypes Core
                 DebugInfoCodeView DebugInfoDWARF DebugInfoDWARFLowLevel
                 MC MCParser Remarks Support Target TargetParser)
endif()

add_llvm_component_library(LLVMAsmPrinter
  ${SOURCES}

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  ${LINK_COMPS}
  )
