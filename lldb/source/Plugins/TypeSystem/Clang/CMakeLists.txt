add_lldb_library(lldbPluginTypeSystemClang PLUGIN
  ImporterBackedASTSource.cpp
  TypeSystemClang.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginExpressionParserClang
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
    lldbPluginObjCRuntime
  CLANG_LIBS
    clangAST
    clangBasic
    clangFrontend
    clangSema
  LINK_COMPONENTS
    Support
)

lldb_tablegen(TypeSystemClangProperties.inc -gen-lldb-property-defs
  SOURCE TypeSystemClangProperties.td
  TARGET LLDBPluginTypeSystemClangPropertiesGen)

lldb_tablegen(TypeSystemClangPropertiesEnum.inc -gen-lldb-property-enum-defs
  SOURCE TypeSystemClangProperties.td
  TARGET LLDBPluginTypeSystemClangPropertiesEnumGen)

add_dependencies(lldbPluginTypeSystemClang
  LLDBPluginTypeSystemClangPropertiesGen
  LLDBPluginTypeSystemClangPropertiesEnumGen)
