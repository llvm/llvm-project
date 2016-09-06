set(LLVM_LINK_COMPONENTS
  ${LLVM_TARGETS_TO_BUILD}
  AsmPrinter
  DebugInfoDWARF
  MC
  Object
  Support
  Target
  )

add_llvm_tool(llvm-dwp
  llvm-dwp.cpp
  DWPError.cpp
  )
