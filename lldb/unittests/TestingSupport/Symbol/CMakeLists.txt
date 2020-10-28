set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbSymbolHelpers
  YAMLModuleTester.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginExpressionParserClang
    lldbPluginSymbolFileDWARF
    lldbPluginTypeSystemClang
    lldbUtilityHelpers

  LINK_COMPONENTS
    ObjectYAML
  )
