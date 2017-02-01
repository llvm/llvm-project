add_lldb_library(lldbPluginLanguageRuntimeJava PLUGIN
  JavaLanguageRuntime.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
