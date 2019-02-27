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
  SwiftASTContext.cpp
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
    swiftAST
    swiftASTSectionImporter
    swiftBasic
    swiftClangImporter
    swiftDriver
    swiftFrontend
    swiftIDE
    swiftSIL
    swiftSerialization
    clangAST
    clangBasic
    clangFrontend
    lldbCore
    lldbExpression
    lldbHost
    lldbTarget
    lldbUtility
    lldbPluginExpressionParserClang
    lldbPluginExpressionParserSwift
    lldbPluginSymbolFileDWARF
    lldbPluginSymbolFilePDB
    lldbPluginCPlusPlusLanguage
    lldbPluginObjCLanguage
    lldbPluginPlatformMacOSX

  LINK_COMPONENTS
    Support
  )
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang AND NOT SWIFT_COMPILER_IS_MSVC_LIKE)
  target_compile_options(lldbSymbol PRIVATE -Wno-dollar-in-identifier-extension)
endif()
