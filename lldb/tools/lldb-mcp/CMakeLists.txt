add_lldb_tool(lldb-mcp
  lldb-mcp.cpp
  Multiplexer.cpp

  LINK_COMPONENTS
    Support
  LINK_LIBS
    liblldb
    lldbHost
    lldbProtocolMCP
    lldbUtility
  )

# lldb-mcp hosts debug sessions in its own process. It drives the debug engine
# through the public SB API in liblldb, but also links a few internal libraries
# (transport, host) directly, so guard against liblldb interposing their
# symbols, the same as lldb-dap.
lldb_prevent_liblldb_symbol_interposition(lldb-mcp)

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
