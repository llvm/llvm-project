add_lldb_library(lldbPluginUnwindAssemblyInstEmulation PLUGIN
  UnwindAssemblyInstEmulation.cpp

  LINK_LIBS
    lldbCore
    lldbRegister
    lldbSymbol
    lldbTarget
  )
