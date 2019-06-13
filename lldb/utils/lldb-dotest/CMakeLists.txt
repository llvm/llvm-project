# Make lldb-dotest a custom target.
add_custom_target(lldb-dotest)
add_dependencies(lldb-dotest ${LLDB_TEST_DEPS})
set_target_properties(lldb-dotest PROPERTIES FOLDER "lldb utils")

get_property(LLDB_DOTEST_ARGS GLOBAL PROPERTY LLDB_DOTEST_ARGS_PROPERTY)

# Generate lldb-dotest Python driver script for each build mode.
if(LLDB_BUILT_STANDALONE)
  set(config_types ".")
  if(CMAKE_CONFIGURATION_TYPES)
    set(config_types ${CMAKE_CONFIGURATION_TYPES})
  endif()
  foreach(config_type ${config_types})
    # In paths to our build-tree, replace CMAKE_CFG_INTDIR with our actual configuration names.
    string(REPLACE ${CMAKE_CFG_INTDIR} ${config_type} config_runtime_output_dir ${LLVM_RUNTIME_OUTPUT_INTDIR})
    string(REPLACE ${LLVM_RUNTIME_OUTPUT_INTDIR} ${config_runtime_output_dir} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")

    # Remaining ones must be paths to the provided LLVM build-tree.
    if(${config_type} IN_LIST LLVM_CONFIGURATION_TYPES)
      # Multi-configuration generator like Xcode (with a matching config).
      string(REPLACE ${CMAKE_CFG_INTDIR} ${config_type} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")
    else()
      # Single-configuration generator like Ninja.
      string(REPLACE ${CMAKE_CFG_INTDIR} "." LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")
    endif()

    configure_file(
      lldb-dotest.in
      ${config_runtime_output_dir}/lldb-dotest @ONLY
    )
  endforeach()
elseif(NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  foreach(LLVM_BUILD_MODE ${CMAKE_CONFIGURATION_TYPES})
    string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_DOTEST_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})
    string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")
    configure_file(
      lldb-dotest.in
      ${LLDB_DOTEST_DIR}/lldb-dotest
      )
  endforeach()
else()
  configure_file(
    lldb-dotest.in
    ${LLVM_RUNTIME_OUTPUT_INTDIR}/lldb-dotest
    )
endif()
