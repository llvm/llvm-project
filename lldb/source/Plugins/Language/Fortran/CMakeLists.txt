add_lldb_library(lldbPluginFortranLanguage PLUGIN
  FortranLanguage.cpp
  DynamicArray.cpp

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbSymbol
    lldbTarget
    lldbUtility
)