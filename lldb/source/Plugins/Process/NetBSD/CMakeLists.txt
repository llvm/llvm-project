add_lldb_library(lldbPluginProcessNetBSD PLUGIN
  NativeProcessNetBSD.cpp
  NativeRegisterContextNetBSD.cpp
  NativeRegisterContextNetBSD_x86_64.cpp
  NativeThreadNetBSD.cpp

  LINK_LIBS
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
  LINK_COMPONENTS
    Support
  )
