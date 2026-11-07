add_lldb_library(lldbPluginRustLanguage PLUGIN
  RustLanguage.cpp
  RustVector.cpp

  LINK_COMPONENTS
    Demangle
    Support
  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
)
