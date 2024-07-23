add_lldb_library(lldbPluginOperatingSystemPythonInterface PLUGIN

  OperatingSystemPythonInterface.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    ${Python3_LIBRARIES}
    ${LLDB_LIBEDIT_LIBS}

  LINK_COMPONENTS
    Support
  )
