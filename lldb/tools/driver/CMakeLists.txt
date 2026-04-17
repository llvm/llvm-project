set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(LLDBOptionsTableGen)

if(APPLE)
  configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lldb-Info.plist.in
    ${CMAKE_CURRENT_BINARY_DIR}/lldb-Info.plist
    )
  # Inline info plist in binary (use target_link_options for this as soon as CMake 3.13 is available)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-sectcreate,__TEXT,__info_plist,${CMAKE_CURRENT_BINARY_DIR}/lldb-Info.plist")

  if(LLDB_CODESIGN_IDENTITY)
    set(LLVM_CODESIGNING_IDENTITY ${LLDB_CODESIGN_IDENTITY})
  elseif(NOT LLVM_CODESIGNING_IDENTITY)
    set(LLVM_CODESIGNING_IDENTITY "-")
  endif()

  # If we're building with assertions and LLDB_ENABLE_MTE is set to ON,
  # build the driver to always run under MTE.
  if (LLDB_ENABLE_MTE AND LLVM_ENABLE_ASSERTIONS)
    set(ENTITLEMENTS ENTITLEMENTS ${CMAKE_CURRENT_SOURCE_DIR}/lldb-mte-entitlements.plist)
  endif()
endif()

if (UNIX AND "${CMAKE_SYSTEM_NAME}" MATCHES "AIX")
  remove_definitions("-D_XOPEN_SOURCE=700")
  add_definitions("-D_ALL_SOURCE")
endif()

set(LLDB_DRIVER_LINK_LIBS
  liblldb
  lldbHost
  lldbUtility
)

if(WIN32)
  list(APPEND LLDB_DRIVER_LINK_LIBS lldbHostPythonPathSetup)
endif()

add_lldb_tool(lldb
  Driver.cpp
  Platform.cpp

  ${ENTITLEMENTS}

  LINK_COMPONENTS
    Option
    Support
  LINK_LIBS
    ${LLDB_DRIVER_LINK_LIBS}
  )

add_dependencies(lldb
  LLDBOptionsTableGen
  ${tablegen_deps}
)

if(LLDB_BUILD_FRAMEWORK)
  # In the build-tree, we know the exact path to the framework directory.
  # The installed framework can be in different locations.
  lldb_setup_rpaths(lldb
    BUILD_RPATH
      "${LLDB_FRAMEWORK_ABSOLUTE_BUILD_DIR}"
    INSTALL_RPATH
      "@loader_path/../../../SharedFrameworks"
      "@loader_path/../../System/Library/PrivateFrameworks"
      "@loader_path/../../Library/PrivateFrameworks"
  )
endif()
