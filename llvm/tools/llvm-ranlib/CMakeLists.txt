set(LLVM_LINK_COMPONENTS archive)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-ranlib
  llvm-ranlib.cpp
  )
