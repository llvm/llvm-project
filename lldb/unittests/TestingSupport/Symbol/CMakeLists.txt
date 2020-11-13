set_property(DIRECTORY PROPERTY EXCLUDE_FROM_ALL ON)
add_lldb_library(lldbSymbolHelpers
  YAMLModuleTester.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbPluginExpressionParserClang
    lldbPluginObjectFileELF
    lldbPluginSymbolFileDWARF
    lldbPluginTypeSystemClang
    lldbUtilityHelpers
    LLVMTestingSupport

  LINK_COMPONENTS
    ObjectYAML
  )
