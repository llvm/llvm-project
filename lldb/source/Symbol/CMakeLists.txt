set(LLVM_OPTIONAL_SOURCES LocateSymbolFileMacOSX.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  set(PLATFORM_SOURCES
    LocateSymbolFileMacOSX.cpp
    )
endif()

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
endif()

add_lldb_library(lldbSymbol
  ArmUnwindInfo.cpp
  Block.cpp
  CompactUnwindInfo.cpp
  CompileUnit.cpp
  CompilerDecl.cpp
  CompilerDeclContext.cpp
  CompilerType.cpp
  DWARFCallFrameInfo.cpp
  DebugMacros.cpp
  DeclVendor.cpp
  FuncUnwinders.cpp
  Function.cpp
  LineEntry.cpp
  LineTable.cpp
  LocateSymbolFile.cpp
  ObjectFile.cpp
  PostfixExpression.cpp
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

  ${SWIFT_SOURCES}
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
    lldbPluginPlatformMacOSX
    ${SWIFT_LIBS}

  LINK_COMPONENTS
    Support
  )
if(CMAKE_CXX_COMPILER_ID STREQUAL Clang AND NOT SWIFT_COMPILER_IS_MSVC_LIKE)
  target_compile_options(lldbSymbol PRIVATE -Wno-dollar-in-identifier-extension)
endif()
