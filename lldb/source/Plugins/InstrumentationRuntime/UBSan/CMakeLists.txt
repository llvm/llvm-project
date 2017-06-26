add_lldb_library(lldbPluginInstrumentationRuntimeUBSan PLUGIN
  UBSanRuntime.cpp

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
