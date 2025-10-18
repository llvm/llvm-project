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
  Protocol.cpp
)

target_compile_features(llvm-lsp-server PRIVATE cxx_std_20)
