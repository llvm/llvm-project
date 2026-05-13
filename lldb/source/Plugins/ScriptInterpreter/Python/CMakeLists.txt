if (APPLE AND LLVM_ENABLE_LOCAL_SUBMODULE_VISIBILITY)
  # Work around an issue with the Python headers, which have a modular include
  # inside an extern "C" block.
  remove_module_flags()
endif()

if(NOT LLDB_PYTHON_RELATIVE_PATH)
  message(FATAL_ERROR "LLDB_PYTHON_RELATIVE_PATH is not set.")
endif()
add_definitions(-DLLDB_PYTHON_RELATIVE_LIBDIR="${LLDB_PYTHON_RELATIVE_PATH}")

if(NOT LLDB_PYTHON_EXE_RELATIVE_PATH)
  message(FATAL_ERROR "LLDB_PYTHON_EXE_RELATIVE_PATH is not set.")
endif()
add_definitions(-DLLDB_PYTHON_EXE_RELATIVE_PATH="${LLDB_PYTHON_EXE_RELATIVE_PATH}")

if (LLDB_ENABLE_LIBEDIT)
  list(APPEND LLDB_LIBEDIT_LIBS LibEdit::LibEdit)
endif()

set(python_plugin_sources
  PythonDataObjects.cpp
  PythonReadline.cpp
  ScriptInterpreterPython.cpp

  Interfaces/OperatingSystemPythonInterface.cpp
  Interfaces/ScriptInterpreterPythonInterfaces.cpp
  Interfaces/ScriptedFramePythonInterface.cpp
  Interfaces/ScriptedFrameProviderPythonInterface.cpp
  Interfaces/ScriptedPlatformPythonInterface.cpp
  Interfaces/ScriptedProcessPythonInterface.cpp
  Interfaces/ScriptedPythonInterface.cpp
  Interfaces/ScriptedStopHookPythonInterface.cpp
  Interfaces/ScriptedHookPythonInterface.cpp
  Interfaces/ScriptedBreakpointPythonInterface.cpp
  Interfaces/ScriptedThreadPlanPythonInterface.cpp
  Interfaces/ScriptedThreadPythonInterface.cpp
)

if (WIN32)
  # Add the library directories to allow Python to use #pragma comment(lib, "python3.lib").
  # FIXME: We should use Python3_SABI_LIBRARY_DIRS here, but this requires CMake 3.26.
  set(PYTHON_SABI_LIBRARY_DIRS ${Python3_LIBRARY_DIRS})
endif()

if (LLDB_ENABLE_DYNAMIC_SCRIPTINTERPRETERS)
  # Shared library loaded at runtime by PluginManager. Private lldb symbols are
  # resolved via liblldb's re-exports, so lldb_private libs aren't linked here.
  add_lldb_library(lldbPluginScriptInterpreterPython SHARED
    ${python_plugin_sources}
    LINK_LIBS
      liblldb
      ${Python3_LIBRARIES}
      ${LLDB_LIBEDIT_LIBS}
  )
  add_python_wrapper(lldbPluginScriptInterpreterPython)

  # Static variant linked directly by unit tests. Separate compilation is
  # required so llvm::Error RTTI (ErrorInfoBase::ID) has a single address
  # shared between the test binary and the plugin code.
  add_lldb_library(lldbStaticScriptInterpreterPython
    ${python_plugin_sources}
    LINK_COMPONENTS
      Support
    LINK_LIBS
      lldbBreakpoint
      lldbCore
      lldbDataFormatters
      lldbHost
      lldbInterpreter
      lldbTarget
      lldbValueObject
      ${Python3_LIBRARIES}
      ${LLDB_LIBEDIT_LIBS}
  )
  target_include_directories(lldbStaticScriptInterpreterPython
    PUBLIC ${Python3_INCLUDE_DIRS})

  target_link_directories(lldbPluginScriptInterpreterPython PRIVATE ${PYTHON_SABI_LIBRARY_DIRS})
  target_link_directories(lldbStaticScriptInterpreterPython PUBLIC ${PYTHON_SABI_LIBRARY_DIRS})
else()
  add_lldb_library(lldbPluginScriptInterpreterPython PLUGIN
    ${python_plugin_sources}
    LINK_COMPONENTS
      Support
    LINK_LIBS
      lldbBreakpoint
      lldbCore
      lldbDataFormatters
      lldbHost
      lldbInterpreter
      lldbTarget
      lldbValueObject
      ${Python3_LIBRARIES}
      ${LLDB_LIBEDIT_LIBS}
  )
  target_link_directories(lldbPluginScriptInterpreterPython PUBLIC ${PYTHON_SABI_LIBRARY_DIRS})
endif()

target_include_directories(lldbPluginScriptInterpreterPython
  PUBLIC ${Python3_INCLUDE_DIRS})
