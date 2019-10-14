# Dbghelp is used on windows for writing minidump files.
if(WIN32)
  set(DBGHELP_LINK_FILES dbghelp)
else()
  set(DBGHELP_LINK_FILES "")
endif()

add_lldb_library(lldbPluginObjectFilePECOFF PLUGIN
  ObjectFilePECOFF.cpp
  PECallFrameInfo.cpp
  WindowsMiniDump.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbSymbol
    lldbTarget
    ${DBGHELP_LINK_FILES}
  LINK_COMPONENTS
    BinaryFormat
    Support
  )
