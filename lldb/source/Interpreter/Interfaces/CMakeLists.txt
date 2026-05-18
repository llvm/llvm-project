add_lldb_library(lldbInterpreterInterfaces NO_PLUGIN_DEPENDENCIES
  ScriptedInterfaceUsages.cpp

  ADDITIONAL_HEADER_DIRS
    ${LLDB_INCLUDE_DIR}/lldb/Interpreter/Interfaces
  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbUtility
  )

