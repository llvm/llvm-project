add_lldb_library(lldbPluginInstrumentationRuntimeASan PLUGIN
  ASanRuntime.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
