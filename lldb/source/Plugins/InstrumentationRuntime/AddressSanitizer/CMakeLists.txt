add_lldb_library(lldbPluginInstrumentationRuntimeAddressSanitizer PLUGIN
  AddressSanitizerRuntime.cpp

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
