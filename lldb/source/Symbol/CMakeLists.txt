set(SWIFT_LIBS
    swiftAST
    swiftASTSectionImporter
    swiftBasic
    swiftClangImporter
    swiftFrontend
    swiftSIL
    swiftSerialization
)
list(APPEND LLVM_OPTIONAL_SOURCES ${SWIFT_SOURCES})
if (NOT LLDB_ENABLE_SWIFT_SUPPORT)
  unset(SWIFT_SOURCES)
  unset(SWIFT_LIBS)
  set(PLUGIN_DEPENDENCY_ARG NO_PLUGIN_DEPENDENCIES)
  set(LLDB_PLUGIN_DEPENDENCIES)
else()
  set(PLUGIN_DEPENDENCY_ARG)
  set(LLDB_PLUGIN_DEPENDENCIES lldbPluginSwiftLanguageRuntime)
endif()

add_lldb_library(lldbSymbol ${PLUGIN_DEPENDENCY_ARG}
  ArmUnwindInfo.cpp
  Block.cpp
  CompactUnwindInfo.cpp
  CompileUnit.cpp
  CompilerDecl.cpp
  CompilerDeclContext.cpp
  CompilerType.cpp
  SaveCoreOptions.cpp
  DWARFCallFrameInfo.cpp
  DebugMacros.cpp
  DeclVendor.cpp
  FuncUnwinders.cpp
  Function.cpp
  ImportedDeclaration.cpp
  LineEntry.cpp
  LineTable.cpp
  ObjectContainer.cpp
  ObjectFile.cpp
  PostfixExpression.cpp
  Symbol.cpp
  SymbolContext.cpp
  SymbolFile.cpp
  SymbolFileOnDemand.cpp
  SymbolLocator.cpp
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

  ## BEGIN SWIFT
  ${SWIFT_SOURCES}
  ## END SWIFT

  ADDITIONAL_HEADER_DIRS
    ${LLDB_INCLUDE_DIR}/lldb/Symbol
  LINK_COMPONENTS
    Support
  LINK_LIBS
    clangAST
    clangBasic
    clangFrontend
    lldbCore
    lldbExpression
    lldbHost
    lldbTarget
    lldbUtility
    lldbValueObject
    ${LLDB_PLUGIN_DEPENDENCIES}
    ## BEGIN SWIFT
    ${SWIFT_LIBS}
    ## END SWIFT

  )
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang AND NOT SWIFT_COMPILER_IS_MSVC_LIKE)
  target_compile_options(lldbSymbol PRIVATE -Wno-dollar-in-identifier-extension)
endif()
