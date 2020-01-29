add_lldb_library(lldbPluginCPPRuntime PLUGIN
  CPPLanguageRuntime.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbPluginTypeSystemClang
)

add_subdirectory(ItaniumABI)
#add_subdirectory(MicrosoftABI)
