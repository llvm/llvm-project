add_llvm_component_library(LLVMAsmPrinter
  AccelTable.cpp
  AddressPool.cpp
  AIXException.cpp
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
  PseudoProbePrinter.cpp
  WinCFGuard.cpp
  WinException.cpp
  CodeViewDebug.cpp
  WasmException.cpp

  DEPENDS
  intrinsics_gen

  LINK_COMPONENTS
  Analysis
  BinaryFormat
  CodeGen
  CodeGenTypes
  Core
  MC
  Remarks
  Support
  Target
  TargetParser
  )

if(EJIT_BARE_METAL)
  target_compile_definitions(LLVMAsmPrinter PRIVATE EJIT_BARE_METAL)
else()
  target_link_libraries(LLVMAsmPrinter PRIVATE
    LLVMDebugInfoCodeView LLVMDebugInfoDWARF LLVMDebugInfoDWARFLowLevel LLVMMCParser)
endif()
