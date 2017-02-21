add_lldb_library(lldbPluginOSGo PLUGIN
  OperatingSystemGo.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  )
