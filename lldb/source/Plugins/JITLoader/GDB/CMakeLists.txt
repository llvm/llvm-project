include_directories(.)

add_lldb_library(lldbPluginJITLoaderGDB PLUGIN
  JITLoaderGDB.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
  LINK_COMPONENTS
    Support
  )

