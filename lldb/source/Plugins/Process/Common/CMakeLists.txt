add_lldb_library(lldbProcessContextSupport NO_PLUGIN_DEPENDENCIES
  RegisterContext/POSIX/RegisterContextPOSIX_arm.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_arm64.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_loongarch64.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_powerpc.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_ppc64le.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_riscv32.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_riscv64.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_s390x.cpp
  RegisterContext/POSIX/RegisterContextPOSIX_x86.cpp

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
