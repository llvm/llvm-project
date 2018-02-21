if (LLDB_BUILD_FRAMEWORK)
# Set the correct rpath to locate libswiftCore
set(CMAKE_EXE_LINKER_FLAGS "-Wl,-rpath \
    -Wl,${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/${LLDB_FRAMEWORK_RESOURCE_DIR}/Swift/macosx")

add_lldb_tool(repl_swift
  main.c
  )

# Ensure we get a fresh, clean module cache for the swift repl run
# before we build repl_swift.

# Note we need to support cmake 2.8, so I could not use named
# add_custom_command entries here.  This will force delete/create
# of the module cache each run of the build.  Not a big deal, but
# if we ever require cmake 3, changing these to add_custom_command
# would do this if repl_swift was out of date.
add_custom_target(
  delete_swift_module_dir
  COMMAND ${CMAKE_COMMAND} -E remove_directory "${module_cache_dir}"
  )

add_custom_target(
  create_swift_module_dir
  COMMAND ${CMAKE_COMMAND} -E make_directory "${module_cache_dir}"
  DEPENDS delete_swift_module_dir
  )

# lldb --repl expects the dummy executable to be located in the framework resource dir,
# so we copy it.
  add_custom_command(TARGET repl_swift POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
            ${CMAKE_BINARY_DIR}/bin/repl_swift
            ${CMAKE_BINARY_DIR}/${LLDB_FRAMEWORK_INSTALL_DIR}/${LLDB_FRAMEWORK_RESOURCE_DIR}/repl_swift
    )
endif()
