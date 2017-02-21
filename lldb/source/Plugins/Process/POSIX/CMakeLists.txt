include_directories(.)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessPOSIX PLUGIN
  CrashReason.cpp
  ProcessMessage.cpp
  ProcessPOSIXLog.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
  LINK_COMPONENTS
    Support
  )
