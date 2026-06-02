add_lldb_library(lldbPluginProcessEZH PLUGIN
  ProcessEZH.cpp
  RegisterContextEZH.cpp
  ThreadEZH.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
    lldbPluginProcessGDBRemote
  LINK_COMPONENTS
    Support
  )
