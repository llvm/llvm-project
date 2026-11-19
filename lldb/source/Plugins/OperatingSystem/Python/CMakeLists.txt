add_lldb_library(lldbPluginOperatingSystemPython PLUGIN
  OperatingSystemPython.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbRegister
    lldbSymbol
    lldbTarget
    lldbValueObject
    lldbPluginProcessUtility
  )
