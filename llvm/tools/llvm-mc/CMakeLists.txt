set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCParser
  Support
  TargetParser
  UnwindInfoChecker
  )

add_llvm_tool(llvm-mc
  llvm-mc.cpp
  Disassembler.cpp
  )

