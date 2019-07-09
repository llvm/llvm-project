set(ALL_LLDB_TEST_EXECUTABLES)

function(add_lldb_test_executable test_name)
  set(EXCLUDE_FROM_ALL ON)
  add_llvm_executable(${test_name} NO_INSTALL_RPATH ${ARGN})
  set(outdir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
  set_output_directory(${test_name} BINARY_DIR ${outdir} LIBRARY_DIR ${outdir})
  list(APPEND ALL_LLDB_TEST_EXECUTABLES ${test_name})
  set(ALL_LLDB_TEST_EXECUTABLES ${ALL_LLDB_TEST_EXECUTABLES} PARENT_SCOPE)
  set_target_properties(${test_name} PROPERTIES FOLDER "lldb tests")
endfunction()

add_lldb_test_executable(thread_inferior inferior/thread_inferior.cpp)
add_lldb_test_executable(environment_check inferior/environment_check.cpp)

if(DEBUGSERVER_PATH)
  add_definitions(-DLLDB_SERVER="${DEBUGSERVER_PATH}" -DLLDB_SERVER_IS_DEBUGSERVER=1)
else()
  add_definitions(-DLLDB_SERVER="$<TARGET_FILE:lldb-server>" -DLLDB_SERVER_IS_DEBUGSERVER=0)
endif()

add_definitions(
  -DLLDB_TEST_INFERIOR_PATH="${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}"
  -DLLDB_TEST_INFERIOR_SUFFIX="${CMAKE_EXECUTABLE_SUFFIX}"
  )
add_subdirectory(tests)
