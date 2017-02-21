add_lldb_library(lldbPluginOSPython PLUGIN
  OperatingSystemPython.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbPluginProcessUtility
  )
