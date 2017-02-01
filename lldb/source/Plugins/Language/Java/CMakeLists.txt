add_lldb_library(lldbPluginJavaLanguage PLUGIN
  JavaFormatterFunctions.cpp
  JavaLanguage.cpp

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
)
