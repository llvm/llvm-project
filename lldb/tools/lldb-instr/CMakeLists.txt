add_lldb_tool(lldb-instr
  Instrument.cpp

  LINK_LIBS
    clangAST
    clangBasic
    clangCodeGen
    clangFrontend
    clangLex
    clangRewrite
    clangSerialization
    clangTooling

  LINK_COMPONENTS
    Support
  )
