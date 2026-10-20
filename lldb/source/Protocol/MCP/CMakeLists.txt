add_lldb_library(lldbProtocolMCP NO_PLUGIN_DEPENDENCIES
  Client.cpp
  MCPError.cpp
  Protocol.cpp
  Server.cpp
  Tool.cpp
  Transport.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbHost
    lldbUtility
)

