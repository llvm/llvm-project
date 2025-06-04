add_lldb_library(lldbInterpreterInterfaces NO_PLUGIN_DEPENDENCIES
  ScriptedInterfaceUsages.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbUtility
  )

