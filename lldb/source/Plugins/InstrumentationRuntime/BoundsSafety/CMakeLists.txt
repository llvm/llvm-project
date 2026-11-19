add_lldb_library(lldbPluginInstrumentationRuntimeBoundsSafety PLUGIN
  InstrumentationRuntimeBoundsSafety.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbRegister
    lldbSymbol
    lldbTarget
    lldbPluginInstrumentationRuntimeUtility

  CLANG_LIBS
    clangCodeGen
  )
