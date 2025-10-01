add_lldb_library(lldbPluginJITLoaderPython PLUGIN
  JITLoaderPython.cpp

  LINK_LIBS
    lldbCore
    lldbInterpreter
    lldbSymbol
    lldbTarget
    lldbValueObject
    lldbPluginProcessUtility
  )
