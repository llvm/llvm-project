add_lldb_library(lldbPluginDynamicLoaderPosixDYLD PLUGIN
  AuxVector.cpp
  DYLDRendezvous.cpp
  DynamicLoaderPOSIXDYLD.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    lldbPluginProcessElfCore
  LINK_COMPONENTS
    Support
  )
