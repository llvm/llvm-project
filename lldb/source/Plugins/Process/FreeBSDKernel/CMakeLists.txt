if (NOT FBSDVMCore_FOUND)
  message(STATUS "Skipping FreeBSDKernel plugin due to missing libfbsdvmcore")
  return()
endif()

add_lldb_library(lldbPluginProcessFreeBSDKernel PLUGIN
  ProcessFreeBSDKernel.cpp
  RegisterContextFreeBSDKernel_arm64.cpp
  RegisterContextFreeBSDKernel_i386.cpp
  RegisterContextFreeBSDKernel_x86_64.cpp
  ThreadFreeBSDKernel.cpp

  LINK_LIBS
    lldbCore
    lldbTarget
    fbsdvmcore
  LINK_COMPONENTS
    Support
  )
