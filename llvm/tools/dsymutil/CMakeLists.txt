set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(DsymutilTableGen)

set(LLVM_LINK_COMPONENTS
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  AsmPrinter
  CodeGen
  DWARFLinker
  DebugInfoDWARF
  MC
  Object
  Option
  Remarks
  Support
  Target
  TargetParser
  )

add_llvm_tool(dsymutil
  dsymutil.cpp
  BinaryHolder.cpp
  CFBundle.cpp
  DebugMap.cpp
  DwarfLinkerForBinary.cpp
  MachODebugMapParser.cpp
  MachOUtils.cpp
  Reproducer.cpp
  SymbolMap.cpp

  DEPENDS
  intrinsics_gen
  ${tablegen_deps}
  DsymutilTableGen
  )

if(APPLE)
  target_link_libraries(dsymutil PRIVATE "-framework CoreFoundation")
endif(APPLE)

# target_link_libraries(dsymutil PRIVATE ${LLVM_ATOMIC_LIB})
