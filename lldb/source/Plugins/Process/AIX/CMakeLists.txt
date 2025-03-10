add_definitions("-D_ALL_SOURCE")

add_lldb_library(lldbPluginProcessAIX
  NativeProcessAIX.cpp
  NativeRegisterContextAIX.cpp
  NativeRegisterContextAIX_ppc64.cpp
  NativeThreadAIX.cpp

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
