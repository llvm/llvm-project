set(derived_headers_location "${CMAKE_CURRENT_BINARY_DIR}/DerivedHeaders")
set(original_headers_location "${LLDB_SOURCE_DIR}/include/lldb")
set(headers_to_process
  API/SBDefines.h
  lldb-defines.h
  lldb-enumerations.h
  lldb-types.h
)

file(MAKE_DIRECTORY ${derived_headers_location})

set(original_headers)
set(derived_headers)
foreach(header ${headers_to_process})
  set(original_header "${original_headers_location}/${header}")

  get_filename_component(header_filename ${header} NAME)
  string(REPLACE "lldb-" "lldb-rpc-" rpc_header_filename "${header_filename}")
  set(derived_header "${derived_headers_location}/${rpc_header_filename}")

  list(APPEND original_headers "${original_header}")
  list(APPEND derived_headers "${derived_header}")
  add_custom_command(OUTPUT ${derived_header}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/convert-lldb-header-to-rpc-header.py
            ${original_header} ${derived_header}
    DEPENDS ${original_header}

    COMMENT "Creating ${derived_header}"
  )
endforeach()

set(generated_headers_to_process
  API/SBLanguages.h
)
foreach(header ${generated_headers_to_process})
  set(original_header "${LLDB_OBJ_DIR}/include/lldb/${header}")

  get_filename_component(header_filename ${header} NAME)
  string(REPLACE "lldb-" "lldb-rpc-" rpc_header_filename "${header_filename}")
  set(derived_header "${derived_headers_location}/${rpc_header_filename}")

  list(APPEND original_headers "${original_header}")
  list(APPEND derived_headers "${derived_header}")
  add_custom_command(OUTPUT ${derived_header}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/convert-lldb-header-to-rpc-header.py
            ${original_header} ${derived_header}
    DEPENDS lldb-sbapi-dwarf-enums

    COMMENT "Creating ${derived_header}"
  )
endforeach()


add_custom_target(copy-aux-rpc-headers DEPENDS ${derived_headers})

set(public_headers ${lldb_rpc_gen_lib_header_files})
list(APPEND public_headers
  ${derived_headers_location}/SBDefines.h
  ${derived_headers_location}/SBLanguages.h
  ${derived_headers_location}/lldb-rpc-enumerations.h
  ${derived_headers_location}/lldb-rpc-types.h
)

# Collect and preprocess headers for the framework bundle
set(version_header
  ${derived_headers_location}/lldb-rpc-defines.h
)

function(FixIncludePaths in subfolder out)
  get_filename_component(base_name ${in} NAME)
  set(parked_header ${CMAKE_CURRENT_BINARY_DIR}/ParkedHeaders/${subfolder}/${base_name})
  set(${out} ${parked_header} PARENT_SCOPE)

  add_custom_command(OUTPUT ${parked_header}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/framework-header-include-fix.py
            ${in} ${parked_header}
    DEPENDS ${in}
    COMMENT "Fixing includes in ${in}"
  )
endfunction()

function(FixVersions in subfolder out)
  get_filename_component(base_name ${in} NAME)
  set(parked_header ${CMAKE_CURRENT_BINARY_DIR}/ParkedHeaders/${subfolder}/${base_name})
  set(${out} ${parked_header} PARENT_SCOPE)

  add_custom_command(OUTPUT ${parked_header}
    COMMAND ${LLDB_SOURCE_DIR}/scripts/framework-header-version-fix.py
            ${in} ${parked_header} ${LLDB_VERSION_MAJOR} ${LLDB_VERSION_MINOR} ${LLDB_VERSION_PATCH}
    DEPENDS ${in}
    COMMENT "Fixing versions in ${liblldbrpc_version_header}"
  )
endfunction()

set(preprocessed_headers)

# Apply include-paths fix on all headers and park them.
foreach(source_header ${public_headers})
  FixIncludePaths(${source_header} Headers parked_header)
  list(APPEND preprocessed_headers ${parked_header})
endforeach()

# Apply include-paths fix and stage in parent directory.
# Then apply version fix and park together with all the others.
FixIncludePaths(${version_header} ".." staged_header)
FixVersions(${staged_header} Headers parked_header)
list(APPEND preprocessed_headers ${parked_header})

# Wrap header preprocessing in a target, so liblldbrpc can depend on.
add_custom_target(liblldbrpc-headers DEPENDS ${preprocessed_headers})
add_dependencies(liblldbrpc-headers copy-aux-rpc-headers)
set_target_properties(liblldbrpc-headers PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ParkedHeaders
)
