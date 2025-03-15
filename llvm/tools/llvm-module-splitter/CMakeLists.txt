set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  LLVMBitReader
  LLVMBitWriter
  LLVMModuleSplitter
  Support
  TransformUtils
)

add_llvm_tool(llvm-module-splitter
  llvm-module-splitter.cpp
)
