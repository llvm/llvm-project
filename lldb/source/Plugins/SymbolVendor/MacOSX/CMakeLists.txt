include_directories(${LIBXML2_INCLUDE_DIR})

add_lldb_library(lldbPluginSymbolVendorMacOSX PLUGIN
  SymbolVendorMacOSX.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
  )
