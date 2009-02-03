set(LLVM_LINK_COMPONENTS ipo scalaropts linker archive bitwriter)
set(LLVM_REQUIRES_EH 1)

add_llvm_tool(llvm-ld
  Optimize.cpp
  llvm-ld.cpp
  )
