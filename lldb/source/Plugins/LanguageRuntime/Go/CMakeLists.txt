set(LLVM_NO_RTTI 1)

add_lldb_library(lldbPluginLanguageRuntimeGo PLUGIN
  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbSymbol
    lldbTarget
  LINK_COMPONENTS
    Support
  )
