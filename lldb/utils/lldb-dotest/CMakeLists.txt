# Make lldb-dotest a custom target.
add_custom_target(lldb-dotest)
add_dependencies(lldb-dotest ${LLDB_TEST_DEPS})

get_property(LLDB_DOTEST_ARGS GLOBAL PROPERTY LLDB_DOTEST_ARGS_PROPERTY)

# Generate wrapper for each build mode.
if(NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
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
