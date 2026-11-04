add_lldb_library(lldbPluginProtocolServerMCP PLUGIN
  ProtocolServerMCP.cpp
  Resource.cpp
  Tool.cpp

  LINK_COMPONENTS
    Support

  LINK_LIBS
    lldbHost
    lldbProtocolMCP
    lldbUtility
)
