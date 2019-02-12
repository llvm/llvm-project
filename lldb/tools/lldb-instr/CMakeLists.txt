add_lldb_tool(lldb-instr
  Instrument.cpp

  LINK_LIBS
    clangAST
    clangBasic
    clangCodeGen
    clangFrontend
    clangTooling

  LINK_COMPONENTS
    Support
  )
