add_lldb_library(lldbProtocolMCP NO_PLUGIN_DEPENDENCIES
  Protocol.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbUtility
)
