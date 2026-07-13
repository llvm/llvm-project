add_lldb_library(lldbPluginObjectFilePlaceholder PLUGIN
  ObjectFilePlaceholder.cpp

  LINK_COMPONENTS
    Object
    Support
  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
  )
