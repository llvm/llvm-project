set(LLVM_LINK_COMPONENTS
  DebugInfo
  Object
  )

add_llvm_tool(llvm-dwarfdump
  llvm-dwarfdump.cpp
  )
