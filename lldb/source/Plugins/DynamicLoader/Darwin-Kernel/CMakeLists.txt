add_lldb_library(lldbPluginDynamicLoaderDarwinKernel PLUGIN
  DynamicLoaderDarwinKernel.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginPlatformMacOSX
  )
