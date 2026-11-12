add_lldb_library(lldbPluginArchitectureWasm PLUGIN
  ArchitectureWasm.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbUtility
  )
