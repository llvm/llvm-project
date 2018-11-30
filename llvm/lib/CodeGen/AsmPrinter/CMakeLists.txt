# Based on whether BPF target is built or not,
# either BTFDebug.cpp or BTFDebugStub.cpp will be included,
# but not both.
set(LLVM_OPTIONAL_SOURCES "BTFDebug.cpp" "BTFDebugStub.cpp")
list(FIND LLVM_TARGETS_TO_BUILD "BPF" idx)
if( NOT idx LESS 0 )
  set(BTFDebug_File "BTFDebug.cpp")
else()
  set(BTFDebug_File "BTFDebugStub.cpp")
endif()

add_llvm_library(LLVMAsmPrinter
  AccelTable.cpp
  AddressPool.cpp
  ARMException.cpp
  AsmPrinter.cpp
  AsmPrinterDwarf.cpp
  AsmPrinterInlineAsm.cpp
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
  EHStreamer.cpp
  ErlangGCPrinter.cpp
  OcamlGCPrinter.cpp
  WinCFGuard.cpp
  WinException.cpp
  CodeViewDebug.cpp
  WasmException.cpp
  ${BTFDebug_File}

  DEPENDS
  intrinsics_gen
  )
