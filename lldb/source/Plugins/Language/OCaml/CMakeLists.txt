add_lldb_library(lldbPluginOCamlLanguage PLUGIN
  OCamlLanguage.cpp

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
)

