set(LLVM_OPTIONAL_SOURCES LocateSymbolFileMacOSX.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(PLATFORM_SOURCES LocateSymbolFileMacOSX.cpp)
endif()

add_lldb_library(lldbSymbol
  ArmUnwindInfo.cpp
  Block.cpp
  ClangASTContext.cpp
  ClangASTImporter.cpp
  ClangExternalASTSourceCallbacks.cpp
  ClangExternalASTSourceCommon.cpp
  ClangUtil.cpp
  CompilerDecl.cpp
  CompilerDeclContext.cpp
  CompilerType.cpp
  CompileUnit.cpp
  CompactUnwindInfo.cpp
  DebugMacros.cpp
  Declaration.cpp
  DWARFCallFrameInfo.cpp
  Function.cpp
  FuncUnwinders.cpp
  LineEntry.cpp
  LineTable.cpp
  LocateSymbolFile.cpp
  ObjectFile.cpp
  Symbol.cpp
  SymbolContext.cpp
  SymbolFile.cpp
  SymbolVendor.cpp
  Symtab.cpp
  Type.cpp
  TypeList.cpp
  TypeMap.cpp
  TypeSystem.cpp
  UnwindPlan.cpp
  UnwindTable.cpp
  Variable.cpp
  VariableList.cpp
  VerifyDecl.cpp

  ${PLATFORM_SOURCES}

  LINK_LIBS
    clangAST
    clangBasic
    clangFrontend
    lldbCore
    lldbExpression
    lldbHost
    lldbTarget
    lldbUtility
    lldbPluginExpressionParserClang
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage

  LINK_COMPONENTS
    Support
  )
