set(LLVM_LINK_COMPONENTS
  AllTargetsAsmPrinters
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  DebugInfoDWARF
  DebugInfoPDB
  MC
  MCDisassembler
  Object
  Support
  Symbolize
  )

add_llvm_tool(sancov
  sancov.cc
  )
