set_property(DIRECTORY PROPERTY LLDB_TOLERATED_PLUGIN_DEPENDENCIES LanguageRuntime)

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
