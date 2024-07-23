add_lldb_library(lldbPluginScriptedThreadPlanPythonInterface PLUGIN

  ScriptedThreadPlanPythonInterface.cpp

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
