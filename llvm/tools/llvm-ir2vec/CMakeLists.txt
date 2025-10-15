set(LLVM_LINK_COMPONENTS
  Analysis
  Core
  IRReader
  Support
  )

add_llvm_tool(llvm-ir2vec
  llvm-ir2vec.cpp
  )
