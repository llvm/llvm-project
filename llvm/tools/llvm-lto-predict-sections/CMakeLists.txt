set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  )

add_llvm_tool(llvm-lto-predict-sections
  llvm-lto-predict-sections.cpp
  )
