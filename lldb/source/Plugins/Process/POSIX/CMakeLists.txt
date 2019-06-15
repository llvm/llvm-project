add_lldb_library(lldbPluginProcessPOSIX PLUGIN
  CrashReason.cpp
  NativeProcessELF.cpp
  ProcessMessage.cpp
  ProcessPOSIXLog.cpp

  LINK_LIBS
    lldbPluginProcessUtility
    lldbUtility
  LINK_COMPONENTS
    Support
  )
