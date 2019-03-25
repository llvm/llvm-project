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

if(NOT LLDB_DISABLE_PYTHON)
  get_target_property(lldb_scripts_dir swig_wrapper BINARY_DIR)
  set(lldb_python_wrapper ${lldb_scripts_dir}/LLDBWrapPython.cpp)
endif()

add_lldb_library(lldbPluginScriptInterpreterPython PLUGIN
  PythonDataObjects.cpp
  PythonExceptionState.cpp
  ScriptInterpreterPython.cpp
  ${lldb_python_wrapper}

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

if(lldb_python_wrapper)
  add_dependencies(lldbPluginScriptInterpreterPython swig_wrapper)

  if (MSVC)
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " /W0")
  else()
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING PROPERTY COMPILE_FLAGS " -w")
  endif()

  set_source_files_properties(${lldb_python_wrapper} PROPERTIES GENERATED ON)
  if (CLANG_CL)
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING
      PROPERTY COMPILE_FLAGS " -Wno-unused-function")
  endif()
  if (LLVM_COMPILER_IS_GCC_COMPATIBLE AND
      NOT "${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
    set_property(SOURCE ${lldb_python_wrapper} APPEND_STRING
      PROPERTY COMPILE_FLAGS " -Wno-sequence-point -Wno-cast-qual")
  endif ()
endif()
