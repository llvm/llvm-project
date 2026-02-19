add_lldb_library(lldbPluginProcessNetBSD
  NativeProcessNetBSD.cpp
  NativeRegisterContextNetBSD.cpp
  NativeRegisterContextNetBSD_x86.cpp
  NativeThreadNetBSD.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
    util
  )
