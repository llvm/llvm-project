set(LLVM_LINK_COMPONENTS ipo bitreader bitwriter)

add_llvm_tool(llvm-extract
  llvm-extract.cpp
  )
