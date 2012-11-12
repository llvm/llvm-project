set(LLVM_LINK_COMPONENTS archive bitreader object)

add_llvm_tool(llvm-nm
  llvm-nm.cpp
  )
