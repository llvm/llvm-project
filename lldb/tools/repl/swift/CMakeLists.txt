set(LLVM_NO_RTTI 1)

include(AddLLVM)
include(SwiftAddCustomCommandTarget)

set(output_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin")
set(module_cache_dir "${CMAKE_BINARY_DIR}/${CMAKE_CFG_INTDIR}/repl_swift_module_cache")

if(NOT LLDB_BUILT_STANDALONE)
  set(swift_depends swift)
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Linux")
  add_custom_command_target(
      repl_swift_target
      COMMAND "${LLDB_PATH_TO_SWIFT_BUILD}/bin/swiftc" -o "${output_dir}/repl_swift" -module-cache-path "${module_cache_dir}" -Xlinker -rpath -Xlinker \$ORIGIN/../lib/swift/linux "${CMAKE_CURRENT_SOURCE_DIR}/main.swift"
      OUTPUT "${output_dir}/repl_swift"
      VERBATIM
      ALL
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/main.swift" ${swift_depends})
else()
  add_custom_command_target(
      repl_swift_target
      COMMAND "${LLDB_PATH_TO_SWIFT_BUILD}/bin/swiftc" -o "${output_dir}/repl_swift" -module-cache-path "${module_cache_dir}" "${CMAKE_CURRENT_SOURCE_DIR}/main.swift"
      OUTPUT "${output_dir}/repl_swift"
      VERBATIM
      ALL
      DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/main.swift" ${swift_depends})
endif()

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
  )

add_dependencies(create_swift_module_dir delete_swift_module_dir)
add_dependencies(${repl_swift_target} create_swift_module_dir)

install(
  PROGRAMS "${output_dir}/repl_swift"
  DESTINATION bin)
