add_lldb_library(lldbPluginCPPRuntime PLUGIN
  CommandObjectCPlusPlus.cpp
  CPPLanguageRuntime.cpp
  ItaniumABIRuntime.cpp
  VerboseTrapFrameRecognizer.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbInterpreter
    lldbPluginTypeSystemClang
    lldbSymbol
    lldbTarget
    lldbValueObject
  CLANG_LIBS
    clangCodeGen
)
