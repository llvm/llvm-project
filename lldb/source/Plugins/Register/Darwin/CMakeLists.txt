add_lldb_library(lldbPluginRegisterDarwin NO_PLUGIN_DEPENDENCIES
  RegisterContextDarwin_arm.cpp
  RegisterContextDarwin_arm64.cpp
  RegisterContextDarwin_riscv32.cpp
  RegisterContextDarwin_x86_64.cpp
  RegisterContextMach_arm.cpp
  RegisterContextMach_x86_64.cpp

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
