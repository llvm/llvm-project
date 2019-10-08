if(NOT LLDB_PYTHON_RELATIVE_PATH)
  message(FATAL_ERROR "LLDB_PYTHON_RELATIVE_PATH is not set.")
endif()
add_definitions(-DLLDB_PYTHON_RELATIVE_LIBDIR="${LLDB_PYTHON_RELATIVE_PATH}")

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
    ${PYTHON_LIBRARY}

  LINK_COMPONENTS
    Support
  )
