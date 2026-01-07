lldb_tablegen(SymbolFilePDBProperties.inc -gen-lldb-property-defs
  SOURCE SymbolFilePDBProperties.td
  TARGET LLDBPluginSymbolFilePDBPropertiesGen)

lldb_tablegen(SymbolFilePDBPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE SymbolFilePDBProperties.td
  TARGET LLDBPluginSymbolFilePDBPropertiesEnumGen)

add_lldb_library(lldbPluginSymbolFilePDB PLUGIN
  PDBASTParser.cpp
  PDBLocationToDWARFExpression.cpp
  SymbolFilePDB.cpp

  LINK_COMPONENTS
    DebugInfoPDB
    Support
  LINK_LIBS
    lldbCore
    lldbPluginSymbolFileNativePDB
    lldbSymbol
    lldbUtility
    lldbPluginTypeSystemClang
  CLANG_LIBS
    clangAST
    clangLex
  )

add_dependencies(lldbPluginSymbolFilePDB
  LLDBPluginSymbolFilePDBPropertiesGen
  LLDBPluginSymbolFilePDBPropertiesEnumGen)
