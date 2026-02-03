# Configure Python runtime library path for Windows
#
# LLDB_PYTHON_DLL_RELATIVE_PATH:
#   Relative path from the LLDB executable to the directory containing the
#   Python runtime DLL. Used on Windows to locate python3X.dll at runtime.
#   Example: "../../../tools/python3/DLLs"
#
# LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME:
#   The filename of the Python runtime library to search for (e.g., "python310.dll").
#   Used on Windows to verify the correct Python DLL is found in the search path.

add_lldb_library(lldbUtilityPythonPathSetup STATIC
  PythonPathSetup.cpp

  LINK_LIBS
    LLVMSupport
)

if(DEFINED LLDB_PYTHON_DLL_RELATIVE_PATH)
  target_compile_definitions(lldbUtilityPythonPathSetup PRIVATE LLDB_PYTHON_DLL_RELATIVE_PATH="${LLDB_PYTHON_DLL_RELATIVE_PATH}")
endif()
if(DEFINED LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME)
  target_compile_definitions(lldbUtilityPythonPathSetup PRIVATE LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME="${LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME}")
endif()
