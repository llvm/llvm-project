if (NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
  # Call a python script to gather the arch-specific libdir for
  # modules like the lldb module.
  execute_process(
    COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../../../scripts/get_relative_lib_dir.py
    RESULT_VARIABLE get_libdir_status
    OUTPUT_VARIABLE relative_libdir
    )
  if (get_libdir_status EQUAL 0)
    add_definitions(-DLLDB_PYTHON_RELATIVE_LIBDIR="${relative_libdir}")
  endif()
endif()

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
