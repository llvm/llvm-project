set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  CAS
  MC
  MCCAS
  MCParser
  Support
  TargetParser
  DebugInfoDWARF
  DWARFCFIChecker
  )

add_llvm_tool(llvm-mc
  llvm-mc.cpp
  Disassembler.cpp
  )

