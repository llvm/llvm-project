add_lldb_library(lldbPluginInstrumentationRuntimeBoundsSafety PLUGIN
  InstrumentationRuntimeBoundsSafety.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginInstrumentationRuntimeUtility

  CLANG_LIBS
    clangCodeGen
  )
