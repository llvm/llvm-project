set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginPlatformMacOSX
  PlatformDarwin.cpp
  PlatformDarwinKernel.cpp
  PlatformiOSSimulator.cpp
  PlatformMacOSX.cpp
  PlatformRemoteiOS.cpp
  )
