add_lldb_library(lldbPluginProcessFreeBSDKernel PLUGIN
  ProcessFreeBSDKernel.cpp
  RegisterContextFreeBSDKernel_arm64.cpp
  RegisterContextFreeBSDKernel_i386.cpp
  RegisterContextFreeBSDKernel_ppc64le.cpp
  RegisterContextFreeBSDKernel_x86_64.cpp
  ThreadFreeBSDKernel.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    kvm
  )
