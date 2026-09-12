add_lldb_library(lldbNativeProcessSupport NO_PLUGIN_DEPENDENCIES
  NativeRegisterContextDBReg.cpp
  NativeRegisterContextDBReg_arm.cpp
  NativeRegisterContextDBReg_arm64.cpp
  NativeRegisterContextDBReg_loongarch.cpp
  NativeRegisterContextDBReg_x86.cpp
  NativeRegisterContextRegisterInfo.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbRegister
    lldbUtility
  ALLOWED_INTERNAL_DEPENDENCIES
    lldbCore
    lldbHost
    lldbRegister
    lldbUtility
  )
