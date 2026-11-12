add_lldb_library(lldbPluginRegisterCore NO_PLUGIN_DEPENDENCIES
  RegisterContextCore_arm.cpp
  RegisterContextCore_arm64.cpp
  RegisterContextCore_loongarch64.cpp
  RegisterContextCore_powerpc.cpp
  RegisterContextCore_ppc64le.cpp
  RegisterContextCore_riscv32.cpp
  RegisterContextCore_riscv64.cpp
  RegisterContextCore_s390x.cpp
  RegisterContextCore_x86.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbRegister
    lldbTarget
    lldbUtility
  ALLOWED_INTERNAL_DEPENDENCIES
    lldbRegister
    lldbTarget
    lldbUtility
  )
