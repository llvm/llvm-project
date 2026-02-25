add_lldb_library(lldbPluginProcessFreeBSDKernelCore PLUGIN
  ProcessFreeBSDKernelCore.cpp
  RegisterContextFreeBSDKernelCore_arm64.cpp
  RegisterContextFreeBSDKernelCore_i386.cpp
  RegisterContextFreeBSDKernelCore_x86_64.cpp
  ThreadFreeBSDKernelCore.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    kvm
  )
