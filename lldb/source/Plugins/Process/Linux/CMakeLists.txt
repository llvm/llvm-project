include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessLinux PLUGIN
  NativeProcessLinux.cpp
  NativeRegisterContextLinux.cpp
  NativeRegisterContextLinux_arm.cpp
  NativeRegisterContextLinux_arm64.cpp
  NativeRegisterContextLinux_x86_64.cpp
  NativeRegisterContextLinux_mips64.cpp
  NativeRegisterContextLinux_s390x.cpp
  NativeThreadLinux.cpp
  ProcessorTrace.cpp
  SingleStepCheck.cpp

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
