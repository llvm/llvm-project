include_directories(.)
include_directories(../POSIX)
include_directories(../Utility)

add_lldb_library(lldbPluginProcessLinux
  NativeProcessLinux.cpp
  NativeRegisterContextLinux.cpp
  NativeRegisterContextLinux_arm.cpp
  NativeRegisterContextLinux_arm64.cpp
  NativeRegisterContextLinux_x86_64.cpp
  NativeRegisterContextLinux_mips64.cpp
  NativeThreadLinux.cpp
  ProcFileReader.cpp
  SingleStepCheck.cpp
  )
