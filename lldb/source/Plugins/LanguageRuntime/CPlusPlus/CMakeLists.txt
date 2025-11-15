add_lldb_library(lldbPluginCPPRuntime
  CPPLanguageRuntime.cpp
  VerboseTrapFrameRecognizer.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
  CLANG_LIBS
    clangCodeGen
)

add_subdirectory(ItaniumABI)
#add_subdirectory(MicrosoftABI)
