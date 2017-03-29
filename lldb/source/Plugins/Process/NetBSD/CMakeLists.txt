include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessNetBSD PLUGIN
  NativeProcessNetBSD.cpp
  NativeRegisterContextNetBSD.cpp
  NativeRegisterContextNetBSD_x86_64.cpp
  NativeThreadNetBSD.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )
