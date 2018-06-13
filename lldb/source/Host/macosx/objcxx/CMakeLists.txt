
remove_module_flags()
include_directories(..)

add_lldb_library(lldbHostMacOSXObjCXX
  Host.mm
  HostInfoMacOSX.mm
  HostThreadMacOSX.mm

  LINK_LIBS
    lldbCore
    lldbSymbol
    lldbTarget
    lldbUtility
    ${LLDB_PLUGINS}
    ${EXTRA_LIBS}

  LINK_COMPONENTS
    Object
    Support
  )
