add_lldb_library(lldbPluginScriptInterpreterPythonScriptedPlatformPythonInterface PLUGIN

  ScriptedPlatformPythonInterface.cpp

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
