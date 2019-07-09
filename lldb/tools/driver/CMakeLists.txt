set(LLVM_TARGET_DEFINITIONS Options.td)
tablegen(LLVM Options.inc -gen-opt-parser-defs)
add_public_tablegen_target(LLDBOptionsTableGen)

if(APPLE)
  # Inline info plist in binary (use target_link_options for this as soon as CMake 3.13 is available)
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-sectcreate,__TEXT,__info_plist,${LLDB_SOURCE_DIR}/tools/driver/lldb-Info.plist")
endif()

add_lldb_tool(lldb
  Driver.cpp
  Platform.cpp

  LINK_LIBS
    liblldb

  LINK_COMPONENTS
    Option
    Support
  )

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  add_definitions( -DIMPORT_LIBLLDB )
endif()

add_dependencies(lldb
  LLDBOptionsTableGen
  ${tablegen_deps}
)

set_target_properties(LLDBOptionsTableGen PROPERTIES FOLDER "lldb misc")

if(LLDB_BUILD_FRAMEWORK)
  # In the build-tree, we know the exact path to the framework directory.
  # The installed framework can be in different locations.
  get_target_property(framework_build_dir liblldb LIBRARY_OUTPUT_DIRECTORY)
  lldb_setup_rpaths(lldb
    BUILD_RPATH
      "${framework_build_dir}"
    INSTALL_RPATH
      "@loader_path/../../../SharedFrameworks"
      "@loader_path/../../System/Library/PrivateFrameworks"
      "@loader_path/../../Library/PrivateFrameworks"
  )
endif()
