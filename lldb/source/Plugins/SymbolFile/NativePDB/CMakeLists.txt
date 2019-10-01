add_lldb_library(lldbPluginSymbolFileNativePDB PLUGIN
  CodeViewRegisterMapping.cpp
  CompileUnitIndex.cpp
  DWARFLocationExpression.cpp
  PdbAstBuilder.cpp
  PdbFPOProgramToDWARFExpression.cpp
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
