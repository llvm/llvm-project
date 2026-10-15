add_lldb_library(lldbPluginDLanguage PLUGIN
  DLanguage.cpp

  LINK_COMPONENTS
    Demangle
    Support
  LINK_LIBS
    lldbCore
    lldbTarget
    lldbUtility
)
