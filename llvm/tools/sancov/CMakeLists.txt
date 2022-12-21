set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  DebugInfoDWARF
  DebugInfoPDB
  MC
  MCDisassembler
  Object
  Support
  TargetParser
  Symbolize
  )

add_llvm_tool(sancov
  sancov.cpp
  )
