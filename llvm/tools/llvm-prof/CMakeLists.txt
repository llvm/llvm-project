set(LLVM_LINK_COMPONENTS bitreader analysis)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-prof
  llvm-prof.cpp
  )
