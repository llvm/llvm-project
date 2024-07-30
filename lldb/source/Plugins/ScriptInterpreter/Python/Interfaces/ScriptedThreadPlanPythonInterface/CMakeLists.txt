add_lldb_library(lldbPluginScriptInterpreterPythonScriptedThreadPlanPythonInterface PLUGIN

  ScriptedThreadPlanPythonInterface.cpp

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
