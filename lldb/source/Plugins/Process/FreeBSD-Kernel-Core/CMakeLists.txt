lldb_tablegen(ProcessFreeBSDKernelCoreProperties.inc -gen-lldb-property-defs
  SOURCE ProcessFreeBSDKernelCoreProperties.td
  TARGET LLDBPluginProcessFreeBSDKernelCorePropertiesGen)

lldb_tablegen(ProcessFreeBSDKernelCorePropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE ProcessFreeBSDKernelCoreProperties.td
  TARGET LLDBPluginProcessFreeBSDKernelCorePropertiesEnumGen)

add_lldb_library(lldbPluginProcessFreeBSDKernelCore PLUGIN
  ProcessFreeBSDKernelCore.cpp
  RegisterContextFreeBSDKernelCore_arm.cpp
  RegisterContextFreeBSDKernelCore_arm64.cpp
  RegisterContextFreeBSDKernelCore_i386.cpp
  RegisterContextFreeBSDKernelCore_ppc64le.cpp
  RegisterContextFreeBSDKernelCore_riscv64.cpp
  RegisterContextFreeBSDKernelCore_x86_64.cpp
  ThreadFreeBSDKernelCore.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    kvm
  )

add_dependencies(lldbPluginProcessFreeBSDKernelCore
  LLDBPluginProcessFreeBSDKernelCorePropertiesGen
  LLDBPluginProcessFreeBSDKernelCorePropertiesEnumGen)
