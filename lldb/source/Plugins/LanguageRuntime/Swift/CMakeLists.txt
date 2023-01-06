add_lldb_library(lldbPluginSwiftLanguageRuntime PLUGIN
  LLDBMemoryReader.cpp
  SwiftLanguageRuntime.cpp
  SwiftLanguageRuntimeDynamicTypeResolution.cpp
  SwiftLanguageRuntimeNames.cpp
  SwiftMetadataCache.cpp

  LINK_LIBS
    swiftAST
    swiftBasic
    swiftRemoteInspection
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
