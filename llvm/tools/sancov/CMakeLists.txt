set(LLVM_LINK_COMPONENTS
  AllTargetsDescs
  AllTargetsDisassemblers
  AllTargetsInfos
  MC
  MCDisassembler
  Object
  Support
  Symbolize
  TargetParser
  )

add_llvm_tool(sancov
  sancov.cpp
  )
