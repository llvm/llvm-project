set(LLVM_TARGET_DEFINITIONS Options.td)
llvm_tablegen(Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(DwarfutilTableGen)

set(LLVM_LINK_COMPONENTS
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  CodeGenTypes
  DWARFLinker
  DWARFLinkerClassic
  DWARFLinkerParallel
  DebugInfoDWARF
  DebugInfoDWARFLowLevel
  MC
  ObjCopy
  Object
  Option
  Support
  Target
  TargetParser
  )

add_llvm_tool(llvm-dwarfutil
  llvm-dwarfutil.cpp
  DebugInfoLinker.cpp

  DEPENDS
  intrinsics_gen
  ${tablegen_deps}
  )
