add_lldb_library(lldbProtocolMCP NO_PLUGIN_DEPENDENCIES
  MCPError.cpp
  Protocol.cpp
  Tool.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbUtility
)

