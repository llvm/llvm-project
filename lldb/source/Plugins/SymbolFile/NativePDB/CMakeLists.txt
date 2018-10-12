add_lldb_library(lldbPluginSymbolFileNativePDB PLUGIN
  CompileUnitIndex.cpp
  PdbIndex.cpp
  PdbUtil.cpp
  SymbolFileNativePDB.cpp

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
