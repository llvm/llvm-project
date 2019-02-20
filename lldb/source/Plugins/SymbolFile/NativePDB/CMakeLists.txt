add_lldb_library(lldbPluginSymbolFileNativePDB PLUGIN
  CompileUnitIndex.cpp
  DWARFLocationExpression.cpp
  PdbAstBuilder.cpp
  PdbIndex.cpp
  PdbSymUid.cpp
  PdbUtil.cpp
  SymbolFileNativePDB.cpp
  UdtRecordCompleter.cpp

  LINK_LIBS
    clangAST
    clangLex
    lldbCore
    lldbSymbol
	  lldbUtility
  LINK_COMPONENTS
    DebugInfoCodeView
    DebugInfoPDB
    Support
  )
