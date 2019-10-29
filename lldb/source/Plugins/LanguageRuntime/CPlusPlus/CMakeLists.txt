add_lldb_library(lldbPluginCPPRuntime PLUGIN
  CPPLanguageRuntime.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
)

add_subdirectory(ItaniumABI)
#add_subdirectory(MicrosoftABI)
