add_lldb_library(lldbInterpreterInterfaces NO_PLUGIN_DEPENDENCIES
  ScriptedInterfaceUsages.cpp

  LINK_LIBS
    lldbUtility

  LINK_COMPONENTS
    Support
  )

