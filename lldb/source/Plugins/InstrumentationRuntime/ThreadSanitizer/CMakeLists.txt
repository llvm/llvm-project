add_lldb_library(lldbPluginInstrumentationRuntimeThreadSanitizer PLUGIN
  ThreadSanitizerRuntime.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  )
