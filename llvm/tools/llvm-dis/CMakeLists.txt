set(LLVM_LINK_COMPONENTS bitreader)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-dis
  llvm-dis.cpp
  )
