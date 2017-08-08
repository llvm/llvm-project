add_lldb_library(lldbPluginInstrumentationRuntimeUndefinedBehaviorSanitizer PLUGIN
  UndefinedBehaviorSanitizerRuntime.cpp

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
