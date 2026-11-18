add_lldb_library(lldbPluginRustLanguage PLUGIN
  RustLanguage.cpp

  LINK_COMPONENTS
    Demangle
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbUtility
)
