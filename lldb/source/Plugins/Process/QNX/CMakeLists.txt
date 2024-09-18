add_lldb_library(lldbPluginProcessQNX
  NativeProcessQNX.cpp
  NativeRegisterContextQNX.cpp
  NativeRegisterContextQNX_arm64.cpp
  NativeThreadQNX.cpp

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
