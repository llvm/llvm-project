set(LLVM_LINK_COMPONENTS asmparser bitwriter)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-as
  llvm-as.cpp
  )
