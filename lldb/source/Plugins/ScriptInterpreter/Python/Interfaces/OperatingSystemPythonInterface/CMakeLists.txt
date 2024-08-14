add_lldb_library(lldbPluginScriptInterpreterPythonOperatingSystemPythonInterface PLUGIN

  OperatingSystemPythonInterface.cpp

  LINK_LIBS
    lldbCore
    lldbHost
    lldbInterpreter
    lldbTarget
    lldbPluginScriptInterpreterPython
    ${Python3_LIBRARIES}
    ${LLDB_LIBEDIT_LIBS}

  LINK_COMPONENTS
    Support
  )
