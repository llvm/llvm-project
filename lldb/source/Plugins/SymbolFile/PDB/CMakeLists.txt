add_lldb_library(lldbPluginSymbolFilePDB PLUGIN
  PDBASTParser.cpp
  PDBLocationToDWARFExpression.cpp
  SymbolFilePDB.cpp

  LINK_LIBS
    clangAST
    clangLex
    lldbCore
    lldbSymbol
	lldbUtility
  LINK_COMPONENTS
    DebugInfoPDB
    Support
  )
