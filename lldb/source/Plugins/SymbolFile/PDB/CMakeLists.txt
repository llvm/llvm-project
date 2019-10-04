add_lldb_library(lldbPluginSymbolFilePDB PLUGIN
  PDBASTParser.cpp
  PDBLocationToDWARFExpression.cpp
  SymbolFilePDB.cpp

  LINK_LIBS
    lldbCore
    lldbPluginSymbolFileNativePDB
    lldbSymbol
    lldbUtility
  CLANG_LIBS
    clangAST
    clangLex
  LINK_COMPONENTS
    DebugInfoPDB
    Support
  )
