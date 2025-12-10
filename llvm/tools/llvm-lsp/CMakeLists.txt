set(LLVM_LINK_COMPONENTS
  Core
  IRReader
  Support
  Analysis
  Passes
  SupportLSP
  TransformUtils
  AsmParser
)

add_llvm_tool(llvm-lsp-server
  llvm-lsp-server.cpp
)
