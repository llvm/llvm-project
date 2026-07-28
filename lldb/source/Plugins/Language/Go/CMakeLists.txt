add_lldb_library(lldbPluginGoLanguage PLUGIN
  GoFormatterFunctions.cpp
  GoLanguage.cpp

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbTarget
)
