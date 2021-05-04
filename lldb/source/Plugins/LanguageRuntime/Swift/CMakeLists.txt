add_lldb_library(lldbPluginSwiftLanguageRuntime PLUGIN
  SwiftLanguageRuntime.cpp
  SwiftLanguageRuntimeDynamicTypeResolution.cpp
  SwiftLanguageRuntimeNames.cpp

  LINK_LIBS
    swiftAST
    swiftBasic
    swiftReflection
    swiftRemoteAST
    lldbCore
    lldbExpression
    lldbInterpreter
    lldbTarget
    lldbUtility
    lldbPluginExpressionParserClang
    lldbPluginProcessUtility
    lldbPluginSwiftLanguage
)
