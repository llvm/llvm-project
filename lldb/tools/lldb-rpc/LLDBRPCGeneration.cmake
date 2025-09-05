if (NOT DEFINED LLDB_RPC_GEN_EXE)
  message(FATAL_ERROR
    "Unable to generate lldb-rpc sources because LLDB_RPC_GEN_EXE is not
    defined. If you are cross-compiling, please build lldb-rpc-gen for your host
    platform.")
endif()
set(lldb_rpc_generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(lldb_rpc_server_generated_source_dir "${lldb_rpc_generated_dir}/server")

file(GLOB api_headers ${LLDB_SOURCE_DIR}/include/lldb/API/SB*.h)
# We don't generate SBCommunication
list(REMOVE_ITEM api_headers ${LLDB_SOURCE_DIR}/include/lldb/API/SBCommunication.h)
# SBDefines.h is mostly definitions and forward declarations, nothing to
# generate.
list(REMOVE_ITEM api_headers ${LLDB_SOURCE_DIR}/include/lldb/API/SBDefines.h)

# Generate the list of byproducts. Note that we cannot just glob the files in
# the directory with the generated sources because BYPRODUCTS needs to be known
# at configure time but the files are generated at build time.
set(lldb_rpc_gen_byproducts
  ${lldb_rpc_generated_dir}/SBClasses.def
  ${lldb_rpc_generated_dir}/SBAPI.def
  ${lldb_rpc_generated_dir}/lldb.py
  ${lldb_rpc_server_generated_source_dir}/SBAPI.h
)

set(lldb_rpc_gen_server_impl_files)
foreach(path ${api_headers})
  get_filename_component(filename_no_ext ${path} NAME_WLE)

  set(server_header_file "Server_${filename_no_ext}.h")
  list(APPEND lldb_rpc_gen_byproducts "${lldb_rpc_server_generated_source_dir}/${server_header_file}")

  set(server_impl_file "Server_${filename_no_ext}.cpp")
  list(APPEND lldb_rpc_gen_byproducts "${lldb_rpc_server_generated_source_dir}/${server_impl_file}")
  list(APPEND lldb_rpc_gen_server_impl_files "${lldb_rpc_server_generated_source_dir}/${server_impl_file}")

endforeach()

# Make sure that the clang-resource-dir is set correctly or else the tool will
# fail to run. This is only needed when we do a standalone build.
set(clang_resource_dir_arg)
if (TARGET clang-resource-headers)
  set(clang_resource_headers_dir
    $<TARGET_PROPERTY:clang-resource-headers,INTERFACE_INCLUDE_DIRECTORIES>)
  set(clang_resource_dir_arg --extra-arg="-resource-dir=${clang_resource_headers_dir}/..")
else()
  set(clang_resource_dir_arg --extra-arg="-resource-dir=${LLDB_EXTERNAL_CLANG_RESOURCE_DIR}")
endif()

set(sysroot_arg)
if (DEFINED TOOLCHAIN_TARGET_SYSROOTFS)
  set(sysroot_arg --extra-arg="-resource-dir=${TOOLCHAIN_TARGET_SYSROOTFS}")
endif()

add_custom_command(OUTPUT ${lldb_rpc_gen_byproducts}
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${lldb_rpc_generated_dir}

  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${lldb_rpc_server_generated_source_dir}

  COMMAND ${LLDB_RPC_GEN_EXE}
    -p ${CMAKE_BINARY_DIR}
    --output-dir=${lldb_rpc_generated_dir}
    ${sysroot_arg}
    --extra-arg="-USWIG"
    ${api_headers}

  DEPENDS ${LLDB_RPC_GEN_EXE} ${api_headers}
  COMMENT "Generating sources for lldb-rpc-server..."
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(lldb-rpc-generate-sources
  DEPENDS
  ${lldb_rpc_gen_byproducts}
  lldb-sbapi-dwarf-enums)

add_dependencies(lldb-rpc-generate-sources clang-resource-headers)
