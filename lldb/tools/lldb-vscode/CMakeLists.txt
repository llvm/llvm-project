if ( CMAKE_SYSTEM_NAME MATCHES "Windows" OR CMAKE_SYSTEM_NAME MATCHES "NetBSD" )
  add_definitions( -DIMPORT_LIBLLDB )
  list(APPEND extra_libs lldbHost)
endif ()

if (HAVE_LIBPTHREAD)
  list(APPEND extra_libs pthread)
endif ()

# We need to include the llvm components we depend on manually, as liblldb does
# not re-export those.
set(LLVM_LINK_COMPONENTS Support)
add_lldb_tool(lldb-vscode
  lldb-vscode.cpp
  BreakpointBase.cpp
  ExceptionBreakpoint.cpp
  FunctionBreakpoint.cpp
  JSONUtils.cpp
  LLDBUtils.cpp
  SourceBreakpoint.cpp
  VSCode.cpp

  LINK_LIBS
    liblldb
    ${host_lib}
    ${extra_libs}

  LINK_COMPONENTS
    Support
  )

if(LLDB_BUILD_FRAMEWORK)
  lldb_setup_framework_rpaths_in_tool(lldb-vscode)
endif()
