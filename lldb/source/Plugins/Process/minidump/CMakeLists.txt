add_lldb_library(lldbPluginProcessMinidump PLUGIN
  MinidumpTypes.cpp
  MinidumpParser.cpp
  RegisterContextMinidump_ARM.cpp
  RegisterContextMinidump_ARM64.cpp
  RegisterContextMinidump_x86.cpp
  ProcessMinidump.cpp
  ThreadMinidump.cpp

  LINK_COMPONENTS
    BinaryFormat
    Object
    Support
  LINK_LIBS
    lldbCore
    lldbPluginObjectFilePlaceholder
    lldbPluginProcessUtility
    lldbPluginRegisterDarwin
    lldbRegister
    lldbTarget
    lldbUtility
  )
