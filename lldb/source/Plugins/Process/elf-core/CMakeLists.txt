include_directories(../Utility)

add_lldb_library(lldbPluginProcessElfCore
  ProcessElfCore.cpp
  ThreadElfCore.cpp
  RegisterContextPOSIXCore_arm.cpp
  RegisterContextPOSIXCore_arm64.cpp
  RegisterContextPOSIXCore_mips64.cpp
  RegisterContextPOSIXCore_powerpc.cpp
  RegisterContextPOSIXCore_s390x.cpp
  RegisterContextPOSIXCore_x86_64.cpp
  )
