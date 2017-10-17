function(add_lldb_test_executable test_name)
  set(EXCLUDE_FROM_ALL ON)
  add_llvm_executable(${test_name} NO_INSTALL_RPATH ${ARGN})
  set(outdir ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR})
  set_output_directory(${test_name} BINARY_DIR ${outdir} LIBRARY_DIR ${outdir})
endfunction()

add_lldb_test_executable(thread_inferior inferior/thread_inferior.cpp)

if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_definitions(-DLLDB_SERVER="$<TARGET_FILE:debugserver>")
else()
  add_definitions(-DLLDB_SERVER="$<TARGET_FILE:lldb-server>")
endif()

add_definitions(-DTHREAD_INFERIOR="${CMAKE_CURRENT_BINARY_DIR}/thread_inferior")
add_subdirectory(tests)
add_dependencies(LLDBServerTests thread_inferior)
