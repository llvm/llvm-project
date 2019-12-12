if(NOT LLDB_PYTHON_RELATIVE_PATH)
  message(FATAL_ERROR "LLDB_PYTHON_RELATIVE_PATH is not set.")
endif()
add_definitions(-DLLDB_PYTHON_RELATIVE_LIBDIR="${LLDB_PYTHON_RELATIVE_PATH}")

if (LLDB_ENABLE_LIBEDIT)
  list(APPEND LLDB_LIBEDIT_LIBS ${libedit_LIBRARIES})
endif()

add_lldb_library(lldbPluginScriptInterpreterPython PLUGIN
  PythonDataObjects.cpp
  PythonReadline.cpp
  ScriptInterpreterPython.cpp

  LINK_LIBS
    lldbBreakpoint
    lldbCore
    lldbDataFormatters
    lldbHost
    lldbInterpreter
    lldbTarget
    ${PYTHON_LIBRARY}
    ${LLDB_LIBEDIT_LIBS}

  LINK_COMPONENTS
    Support
  )

if (LLDB_ENABLE_LIBEDIT)
  target_include_directories(lldbPluginScriptInterpreterPython PUBLIC
     ${libedit_INCLUDE_DIRS}
  )
endif()
