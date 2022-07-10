set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(DwarfutilTableGen)

set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  DWARFLinker
  MC
  ObjCopy
  Object
  Option
  Support
  Target
  AllTargetsCodeGens
  AllTargetsDescs
  AllTargetsInfos
  )

add_llvm_tool(llvm-dwarfutil
  llvm-dwarfutil.cpp
  DebugInfoLinker.cpp

  DEPENDS
  intrinsics_gen
  ${tablegen_deps}
  )
