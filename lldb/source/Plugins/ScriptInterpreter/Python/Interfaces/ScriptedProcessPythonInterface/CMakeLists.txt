add_lldb_library(lldbPluginScriptInterpreterPythonScriptedProcessPythonInterface PLUGIN

  ScriptedProcessPythonInterface.cpp

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
