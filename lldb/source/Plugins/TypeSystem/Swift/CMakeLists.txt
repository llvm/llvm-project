add_lldb_library(lldbPluginTypeSystemSwift PLUGIN
  TypeSystemSwift.cpp
  TypeSystemSwiftTypeRef.cpp
  SwiftASTContext.cpp

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbUtility

  LINK_COMPONENTS
    Support
)
