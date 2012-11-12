set(LLVM_LINK_COMPONENTS bitreader analysis)

add_llvm_tool(llvm-dis
  llvm-dis.cpp
  )
