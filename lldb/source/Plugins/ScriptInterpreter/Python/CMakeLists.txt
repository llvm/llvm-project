add_lldb_library(lldbPluginScriptInterpreterPython PLUGIN
  PythonDataObjects.cpp
  PythonExceptionState.cpp
  ScriptInterpreterPython.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbHost
    lldbInterpreter
    lldbTarget
  LINK_COMPONENTS
    Support
  )
