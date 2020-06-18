set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(DsymutilTableGen)

set(LLVM_LINK_COMPONENTS
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
  MachODebugMapParser.cpp
  MachOUtils.cpp
  Reproducer.cpp
  SymbolMap.cpp

  DEPENDS
  intrinsics_gen
  ${tablegen_deps}
  )

if(APPLE)
  target_link_libraries(dsymutil PRIVATE "-framework CoreFoundation")
endif(APPLE)

if(HAVE_CXX_ATOMICS_WITH_LIB OR HAVE_CXX_ATOMICS64_WITH_LIB)
  target_link_libraries(dsymutil PRIVATE atomic)
endif()
