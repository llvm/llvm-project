set(LLVM_LINK_COMPONENTS
  AllTargetsAsmParsers
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCCFIAnalysis
  MCParser
  Support
  TargetParser
  )

add_llvm_tool(llvm-mc
  llvm-mc.cpp
  Disassembler.cpp
  )

