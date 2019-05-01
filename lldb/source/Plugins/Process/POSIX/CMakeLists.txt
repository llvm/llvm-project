add_lldb_library(lldbPluginProcessPOSIX PLUGIN
  CrashReason.cpp
  ProcessMessage.cpp
  ProcessPOSIXLog.cpp

  LINK_LIBS
    lldbUtility
  LINK_COMPONENTS
    Support
  )
