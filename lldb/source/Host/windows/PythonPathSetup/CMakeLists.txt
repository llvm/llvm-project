add_lldb_library(lldbHostPythonPathSetup STATIC
  PythonPathSetup.cpp
)

if(DEFINED LLDB_PYTHON_DLL_RELATIVE_PATH)
  target_compile_definitions(lldbHostPythonPathSetup PRIVATE LLDB_PYTHON_DLL_RELATIVE_PATH="${LLDB_PYTHON_DLL_RELATIVE_PATH}")
endif()
if(DEFINED LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME)
  target_compile_definitions(lldbHostPythonPathSetup PRIVATE LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME="${LLDB_PYTHON_RUNTIME_LIBRARY_FILENAME}")
endif()
# BEGIN SWIFT
if((DEFINED Python3_VERSION_MAJOR) AND (DEFINED Python3_VERSION_MINOR))
  target_compile_definitions(lldbHostPythonPathSetup PRIVATE LLDB_PYTHON_VERSION="${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}")
endif()
# END SWIFT
