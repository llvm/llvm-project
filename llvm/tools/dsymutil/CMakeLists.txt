set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(DsymutilTableGen)

set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AsmPrinter
  DebugInfoDWARF
  DWARFLinker
  MC
  Object
  CodeGen
  Option
  Remarks
  Support
  Target
  )

add_llvm_tool(dsymutil
  dsymutil.cpp
  BinaryHolder.cpp
  CFBundle.cpp
  DebugMap.cpp
  DwarfLinkerForBinary.cpp
  DwarfStreamer.cpp
  MachODebugMapParser.cpp
  MachOUtils.cpp
  SymbolMap.cpp

  DEPENDS
  intrinsics_gen
  ${tablegen_deps}
  )

if(APPLE)
  target_link_libraries(dsymutil PRIVATE "-framework CoreFoundation")
endif(APPLE)
