## BEGIN SWIFT
set(SWIFT_SOURCES PdbAstBuilderSwift.cpp)
set(LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
endif()
## END SWIFT

add_lldb_library(lldbPluginSymbolFileNativePDB
  CodeViewRegisterMapping.cpp
  CompileUnitIndex.cpp
  DWARFLocationExpression.cpp
  PdbAstBuilderClang.cpp
  PdbFPOProgramToDWARFExpression.cpp
  PdbIndex.cpp
  PdbSymUid.cpp
  PdbUtil.cpp
  SymbolFileNativePDB.cpp
  UdtRecordCompleter.cpp
  ## BEGIN SWIFT
  ${SWIFT_SOURCES}
  ##END SWIFT

  LINK_COMPONENTS
    DebugInfoCodeView
    DebugInfoPDB
    Support
    TargetParser
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbUtility
    lldbPluginTypeSystemClang
    lldbPluginObjectFilePDB
  CLANG_LIBS
    clangAST
    clangLex
  )
