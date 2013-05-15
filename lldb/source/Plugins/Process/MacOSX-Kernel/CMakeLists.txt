set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginProcessMacOSXKernel
  CommunicationKDP.cpp
  ProcessKDP.cpp
  ProcessKDPLog.cpp
  RegisterContextKDP_arm.cpp
  RegisterContextKDP_i386.cpp
  RegisterContextKDP_x86_64.cpp
  ThreadKDP.cpp
  )
