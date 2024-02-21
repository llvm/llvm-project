if(NOT LLDB_BUILT_STANDALONE)
  set(tablegen_deps intrinsics_gen)
endif()

add_lldb_library(lldbPluginExpressionParserSwift PLUGIN
  SwiftASTManipulator.cpp
  SwiftExpressionParser.cpp
  SwiftExpressionSourceCode.cpp
  SwiftExpressionVariable.cpp
  SwiftPersistentExpressionState.cpp
  SwiftREPL.cpp
  SwiftREPLMaterializer.cpp
  SwiftUserExpression.cpp
  SwiftSILManipulator.cpp

  DEPENDS
  ${tablegen_deps}

  LINK_LIBS
    lldbCore
    lldbDataFormatters
    lldbExpression
    lldbHost
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbUtility
    lldbPluginTypeSystemSwift
    lldbPluginSwiftLanguageRuntime
    swiftAST
    swiftBasic
    swiftClangImporter
    swiftFrontend
    swiftIDE
    swiftParse
    swiftSIL
    swiftSILOptimizer
    swiftSerialization
    clangAST
    clangBasic
    clangRewrite

  LINK_COMPONENTS
    Support
    Core
  )

if(BOOTSTRAPPING_MODE)
  target_link_libraries(lldbPluginExpressionParserSwift
    PRIVATE
      swiftCompilerModules)

  if (CMAKE_SYSTEM_NAME MATCHES "Linux|Android|OpenBSD|FreeBSD|Windows")
    # The swiftCompilerModules need swiftrt to work properly.
    string(REGEX MATCH "^[^-]*" arch ${LLVM_TARGET_TRIPLE})
    string(TOLOWER ${CMAKE_SYSTEM_NAME} platform)
    target_link_libraries(lldbPluginExpressionParserSwift
      PRIVATE
        "${LLDB_SWIFT_LIBS}/${platform}/${arch}/swiftrt${CMAKE_C_OUTPUT_EXTENSION}")
  endif()
else()
  target_link_libraries(lldbPluginExpressionParserSwift
    PRIVATE
      swiftCompilerStub)
endif()
