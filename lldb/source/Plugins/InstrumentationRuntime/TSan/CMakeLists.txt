add_lldb_library(lldbPluginInstrumentationRuntimeTSan PLUGIN
  TSanRuntime.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  )
