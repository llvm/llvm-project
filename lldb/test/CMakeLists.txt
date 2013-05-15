function(add_python_test_target name test_script args comment)
  set(PYTHON_TEST_COMMAND
    ${PYTHON_EXECUTABLE}
    ${test_script}
    ${args}
    )

  add_custom_target(${name}
    COMMAND ${PYTHON_TEST_COMMAND} ${ARG_DEFAULT_ARGS}
    COMMENT "${comment}"
    )
endfunction()

string(REGEX REPLACE ".*ccache\ +" "" LLDB_TEST_COMPILER ${CMAKE_C_COMPILER} ${CMAKE_C_COMPILER_ARG1})

# Users can override LLDB_TEST_ARGS to modify the way LLDB tests are run. See help below.
set(LLDB_TEST_ARGS
  -C
  ${LLDB_TEST_COMPILER}
  CACHE STRING "Specify compiler(s) and architecture(s) with which run LLDB tests. For example: '-C gcc -C clang -A i386 -A x86_64'"
  )
string(REPLACE " " ";" LLDB_TEST_ARGS ${LLDB_TEST_ARGS})

set(LLDB_TRACE_DIR "${CMAKE_BINARY_DIR}/lldb-test-traces"
  CACHE STRING "Set directory to output LLDB test traces (for tests that do not pass.)"
  )

set(LLDB_COMMON_TEST_ARGS
  --executable
  ${CMAKE_BINARY_DIR}/bin/lldb
  -s
  ${LLDB_TRACE_DIR}
  -u CXXFLAGS
  -u CFLAGS
  )

add_python_test_target(check-lldb-single
  ${LLDB_SOURCE_DIR}/test/dotest.py
  "${LLDB_COMMON_TEST_ARGS};${LLDB_TEST_ARGS}"
  "Testing LLDB with args: ${LLDB_COMMON_TEST_ARGS};${LLDB_TEST_ARGS}"
  )

set(LLDB_DOSEP_ARGS
  -o;\"-q;${LLDB_COMMON_TEST_ARGS};${LLDB_TEST_ARGS}\"
  )

# If tests crash cause LLDB to crash, or things are otherwise unstable, or if machine-parsable
# output is desired (i.e. in continuous integration contexts) check-lldb-sep is a better target.
add_python_test_target(check-lldb
  ${LLDB_SOURCE_DIR}/test/dosep.ty
  "${LLDB_DOSEP_ARGS}"
  "Testing LLDB (with a separate subprocess per test)"
  )
