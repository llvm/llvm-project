add_lldb_library(lldbPluginCXXMicrosoftABI PLUGIN
  MicrosoftABILanguageRuntime.cpp

  LINK_LIBS
    lldbPluginCPPRuntime
)
