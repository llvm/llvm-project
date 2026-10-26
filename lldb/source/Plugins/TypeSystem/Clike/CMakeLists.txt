add_lldb_library(lldbPluginTypeSystemClike PLUGIN
  Identifier.cpp
  LanguageOpts.cpp
  Type.cpp
  TypeSystemClike.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbUtility
  CLANG_LIBS
    clangBasic
)
