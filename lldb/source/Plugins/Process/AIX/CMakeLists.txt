add_lldb_library(lldbPluginProcessAIX
  NativeProcessAIX.cpp
  NativeThreadAIX.cpp
  NativeRegisterContextAIX.cpp
  NativeRegisterContextAIX_ppc64.cpp
  RegisterLayoutAIX_ppc64.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginProcessPOSIX
    lldbPluginProcessUtility
    lldbPluginRegisterNative
    lldbRegister
    lldbSymbol
    lldbTarget
    lldbUtility
  )

target_compile_definitions(lldbPluginProcessAIX PRIVATE "-D_ALL_SOURCE")
