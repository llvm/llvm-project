set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  LLVMModuleSplitter
  Support
)

add_llvm_tool(llvm-module-splitter
  llvm-module-splitter.cpp

)
