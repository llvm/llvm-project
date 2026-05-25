set(ALL_SOURCES
  AIXException.cpp ARMException.cpp AsmPrinter.cpp AsmPrinterDwarf.cpp
  AsmPrinterInlineAsm.cpp EHStreamer.cpp ErlangGCPrinter.cpp
  OcamlGCPrinter.cpp WasmException.cpp
  AccelTable.cpp AddressPool.cpp CodeViewDebug.cpp
  DbgEntityHistoryCalculator.cpp DebugHandlerBase.cpp DebugLocStream.cpp
  DIE.cpp DIEHash.cpp DwarfCFIException.cpp DwarfCompileUnit.cpp
  DwarfDebug.cpp DwarfExpression.cpp DwarfFile.cpp DwarfStringPool.cpp
  DwarfUnit.cpp PseudoProbePrinter.cpp WinCFGuard.cpp WinException.cpp
)

# When EJIT_BARE_METAL, exclude debug-info LINK_COMPONENTS.
# All source files are still compiled (clang needs them for CFI
# generation), but the JIT never calls DwarfDebug/CodeView/Win
# constructors (guarded in AsmPrinter.cpp), so gc-sections strips them.
set(ASMP_LINK_COMPS_DEBUG DebugInfoCodeView DebugInfoDWARF DebugInfoDWARFLowLevel MCParser)

if(EJIT_BARE_METAL)
  set(LINK_COMPS Analysis BinaryFormat CodeGen CodeGenTypes Core
                 MC Remarks Support Target TargetParser)
else()
  set(LINK_COMPS Analysis BinaryFormat CodeGen CodeGenTypes Core
                 ${ASMP_LINK_COMPS_DEBUG} MC Remarks Support Target TargetParser)
endif()

add_llvm_component_library(LLVMAsmPrinter
  ${ALL_SOURCES}
  DEPENDS intrinsics_gen
  LINK_COMPONENTS ${LINK_COMPS}
  )

if(EJIT_BARE_METAL)
  target_compile_definitions(LLVMAsmPrinter PRIVATE EJIT_BARE_METAL)
endif()
