set(LLVM_LINK_COMPONENTS
  IRReader
  BitWriter
  Core
  Support
  )

add_llvm_tool(llvm-modextract
  llvm-modextract.cpp
  )
