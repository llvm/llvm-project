set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  DebugInfoDWARF
  DebugInfoGSYM
  AsmPrinter
  AllTargetsDescs
  AllTargetsInfos
  MC
  Object
  Support
  Target
  )

add_llvm_tool(llvm-gsymutil
  llvm-gsymutil.cpp

  DEPENDS
  intrinsics_gen
  )
