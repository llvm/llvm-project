add_lldb_library(lldbPluginPlatformFreeBSDKernel PLUGIN
  PlatformFreeBSDKernel.cpp
  TrapframeUnwindPlan_arm64.cpp
  TrapframeUnwindPlan_ppc64le.cpp
  TrapframeUnwindPlan_riscv64.cpp
  TrapframeUnwindPlan_x86_64.cpp

  LINK_COMPONENTS
    TargetParser
   LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbTarget
  )
