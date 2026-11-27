add_lldb_library(lldbPluginProcessElfCore PLUGIN
  ProcessElfCore.cpp
  ThreadElfCore.cpp
  RegisterContextLinuxCore_x86.cpp
  RegisterContextELFCore_arm.cpp
  RegisterContextELFCore_arm64.cpp
  RegisterContextELFCore_powerpc.cpp
  RegisterContextELFCore_ppc64le.cpp
  RegisterContextELFCore_s390x.cpp
  RegisterContextELFCore_x86.cpp
  RegisterContextELFCore_riscv32.cpp
  RegisterContextELFCore_riscv64.cpp
  RegisterContextELFCore_loongarch64.cpp
  RegisterUtilities.cpp

  LINK_COMPONENTS
    BinaryFormat
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbPluginDynamicLoaderPosixDYLD
    lldbPluginObjectFilePlaceholder
    lldbPluginObjectFileELF
    lldbPluginProcessUtility
    lldbPluginRegisterCore
  )
