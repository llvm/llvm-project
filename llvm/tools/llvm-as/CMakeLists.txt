set(LLVM_LINK_COMPONENTS
  Analysis
  AsmParser
  BitWriter
  Core
  Support
  )

add_llvm_tool(llvm-as
  llvm-as.cpp
  )
