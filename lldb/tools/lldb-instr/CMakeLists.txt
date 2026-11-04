add_lldb_tool(lldb-instr
  Instrument.cpp

  LINK_COMPONENTS
    Support
  CLANG_LIBS
    clangAST
    clangBasic
    clangCodeGen
    clangFrontend
    clangLex
    clangRewrite
    clangSerialization
    clangTooling
  )
