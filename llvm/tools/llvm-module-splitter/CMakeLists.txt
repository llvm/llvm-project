set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  BitReader
  BitWriter
  ModuleSplitter
  Support
  TransformUtils
)

add_llvm_tool(llvm-module-splitter
  llvm-module-splitter.cpp
)
