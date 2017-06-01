set(LLVM_LINK_COMPONENTS
  IRReader
  BitReader
  BitWriter
  Core
  Support
  )

add_llvm_tool(llvm-modextract
  llvm-modextract.cpp
  )
