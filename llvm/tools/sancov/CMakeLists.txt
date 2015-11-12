set(LLVM_LINK_COMPONENTS
  DebugInfoDWARF
  DebugInfoPDB
  Object
  Support
  Symbolize
  )

add_llvm_tool(sancov
  sancov.cc
  )
