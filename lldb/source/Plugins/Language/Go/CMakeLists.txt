add_lldb_library(lldbPluginGoLanguage PLUGIN
  GoLanguage.cpp
  GoFormatterFunctions.cpp

  LINK_LIBS
    clangAST
    lldbCore
    lldbDataFormatters
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
)
