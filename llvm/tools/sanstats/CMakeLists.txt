set(LLVM_LINK_COMPONENTS
  Support
  Symbolize
  )

add_llvm_tool(sanstats
  sanstats.cpp
  )
