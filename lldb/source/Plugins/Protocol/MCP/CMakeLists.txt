add_lldb_library(lldbPluginProtocolServerMCP PLUGIN
  MCPError.cpp
  Protocol.cpp
  ProtocolServerMCP.cpp
  Resource.cpp
  Tool.cpp

  LINK_COMPONENTS
    Support

  LINK_LIBS
    lldbHost
    lldbUtility
)
