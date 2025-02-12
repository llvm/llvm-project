add_lldb_library(lldbPluginOperatingSystemSwiftTasks PLUGIN
  OperatingSystemSwiftTasks.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbValueObject
    lldbPluginProcessUtility
    lldbPluginSwiftLanguageRuntime
  )
