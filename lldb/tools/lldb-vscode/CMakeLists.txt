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
  IOStream.cpp
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
  # In the build-tree, we know the exact path to the framework directory.
  # The installed framework can be in different locations.
  get_target_property(framework_build_dir liblldb LIBRARY_OUTPUT_DIRECTORY)
  lldb_setup_rpaths(lldb-vscode
    BUILD_RPATH
      "${framework_build_dir}"
    INSTALL_RPATH
      "@loader_path/../../../SharedFrameworks"
      "@loader_path/../../System/Library/PrivateFrameworks"
      "@loader_path/../../Library/PrivateFrameworks"
  )
endif()
