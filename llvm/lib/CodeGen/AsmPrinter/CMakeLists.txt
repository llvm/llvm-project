set(ASMPRINTER_SOURCES
  AsmPrinter.cpp
  AsmPrinterInlineAsm.cpp
  ErlangGCPrinter.cpp
  OcamlGCPrinter.cpp
  PseudoProbePrinter.cpp
  )

set(ASMPRINTER_DEBUG_SOURCES
  AccelTable.cpp
  AddressPool.cpp
  AsmPrinterDwarf.cpp
  DbgEntityHistoryCalculator.cpp
  DebugHandlerBase.cpp
  DebugLocStream.cpp
  DIE.cpp
  DIEHash.cpp
  DwarfCompileUnit.cpp
  DwarfDebug.cpp
  DwarfExpression.cpp
  DwarfFile.cpp
  DwarfStringPool.cpp
  DwarfUnit.cpp
  )

set(ASMPRINTER_EH_BASE
  EHStreamer.cpp
  )

set(ASMPRINTER_EH_OPT
  AIXException.cpp
  ARMException.cpp
  DwarfCFIException.cpp
  WinException.cpp
  WasmException.cpp
  )

set(ASMPRINTER_LINK
  Analysis
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  MC
  MCParser
  Remarks
  Support
  Target
  TargetParser
  )

set(ASMPRINTER_DEBUG_LINK
  DebugInfoDWARF
  DebugInfoDWARFLowLevel
  )

# EHStreamer base class is always compiled (needed for unique_ptr<E> destructors)
list(APPEND ASMPRINTER_SOURCES ${ASMPRINTER_EH_BASE})

if(NOT EJIT_BARE_METAL AND NOT EJIT_TRIM_LLVM_BACKEND)
  list(APPEND ASMPRINTER_SOURCES ${ASMPRINTER_DEBUG_SOURCES})
  list(APPEND ASMPRINTER_SOURCES ${ASMPRINTER_EH_OPT})
  list(APPEND ASMPRINTER_LINK ${ASMPRINTER_DEBUG_LINK})
  list(APPEND ASMPRINTER_SOURCES CodeViewDebug.cpp)
  list(APPEND ASMPRINTER_SOURCES WinCFGuard.cpp)
  list(APPEND ASMPRINTER_LINK DebugInfoCodeView)
endif()

add_llvm_component_library(LLVMAsmPrinter
  ${ASMPRINTER_SOURCES}

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  ${ASMPRINTER_LINK}
  )
