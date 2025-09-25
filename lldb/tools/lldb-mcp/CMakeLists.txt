add_lldb_tool(lldb-mcp
  lldb-mcp.cpp

  LINK_COMPONENTS
    Option
    Support
  LINK_LIBS
    liblldb
    lldbInitialization
    lldbHost
    lldbProtocolMCP
  )

if(APPLE)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lldb-mcp-Info.plist.in
    ${CMAKE_CURRENT_BINARY_DIR}/lldb-mcp-Info.plist
    )
  target_link_options(lldb-mcp
    PRIVATE LINKER:-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_BINARY_DIR}/lldb-mcp-Info.plist)
endif()

if(LLDB_BUILD_FRAMEWORK)
  # In the build-tree, we know the exact path to the framework directory.
  # The installed framework can be in different locations.
  lldb_setup_rpaths(lldb-mcp
    BUILD_RPATH
      "${LLDB_FRAMEWORK_ABSOLUTE_BUILD_DIR}"
    INSTALL_RPATH
      "@loader_path/../../../SharedFrameworks"
      "@loader_path/../../System/Library/PrivateFrameworks"
      "@loader_path/../../Library/PrivateFrameworks"
  )
endif()
