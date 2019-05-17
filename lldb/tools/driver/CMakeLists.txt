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

if(LLDB_BUILD_FRAMEWORK)
  lldb_setup_framework_rpaths_in_tool(lldb)
endif()
