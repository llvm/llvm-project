function(add_python_test_target name test_script args comment)
  set(PYTHON_TEST_COMMAND
    ${PYTHON_EXECUTABLE}
    ${test_script}
    ${args}
    )

  add_custom_target(${name}
    COMMAND ${PYTHON_TEST_COMMAND} ${ARG_DEFAULT_ARGS}
    COMMENT "${comment}"
    USES_TERMINAL
    )
  add_dependencies(${name} lldb-test-deps)
endfunction()

# The default architecture with which to compile test executables is the default LLVM target
# architecture, which itself defaults to the host architecture.
string(TOLOWER "${LLVM_TARGET_ARCH}" LLDB_DEFAULT_TEST_ARCH)
if( LLDB_DEFAULT_TEST_ARCH STREQUAL "host" )
  string(REGEX MATCH "^[^-]*" LLDB_DEFAULT_TEST_ARCH ${LLVM_HOST_TRIPLE})
endif ()

# Allow the user to override the default by setting LLDB_TEST_ARCH
set(LLDB_TEST_ARCH
  ${LLDB_DEFAULT_TEST_ARCH}
  CACHE STRING "Specify the architecture to run LLDB tests as (x86|x64).  Determines whether tests are compiled with -m32 or -m64")

# Users can override LLDB_TEST_USER_ARGS to specify arbitrary arguments to pass to the script
set(LLDB_TEST_USER_ARGS
  ""
  CACHE STRING "Specify additional arguments to pass to test runner. For example: '-C gcc -C clang -A i386 -A x86_64'")

# The .noindex suffix is a marker for Spotlight to never index the
# build directory.  LLDB queries Spotlight to locate .dSYM bundles
# based on the UUID embedded in a binary, and because the UUID is a
# hash of filename and .text section, there *will* be conflicts inside
# the build directory.
set(LLDB_TEST_COMMON_ARGS
  --arch=${LLDB_TEST_ARCH}
  -s
  ${CMAKE_BINARY_DIR}/lldb-test-traces
  -S nm
  -u CXXFLAGS
  -u CFLAGS
  )

list(APPEND LLDB_TEST_COMMON_ARGS
  --executable ${LLDB_TEST_EXECUTABLE}
  --dsymutil ${LLDB_TEST_DSYMUTIL}
  --filecheck ${LLDB_TEST_FILECHECK}
  -C ${LLDB_TEST_C_COMPILER}
  )

if ( CMAKE_SYSTEM_NAME MATCHES "Windows" )
  # All tests are currently flaky on Windows, so rerun them all once when they fail.
  set(LLDB_TEST_COMMON_ARGS ${LLDB_TEST_COMMON_ARGS} --rerun-all-issues)

  set(LLDB_TEST_DEBUG_TEST_CRASHES
    0
    CACHE BOOL "(Windows only) Enables debugging of tests in the test suite by showing the crash dialog when lldb crashes")

  set(LLDB_TEST_HIDE_CONSOLE_WINDOWS
    1
    CACHE BOOL "(Windows only) Hides the console window for an inferior when it is launched through the test suite")

  if (LLDB_TEST_DEBUG_TEST_CRASHES)
    set(LLDB_TEST_COMMON_ARGS ${LLDB_TEST_COMMON_ARGS} --enable-crash-dialog)
  endif()

  if (NOT LLDB_TEST_HIDE_CONSOLE_WINDOWS)
    set(LLDB_TEST_COMMON_ARGS ${LLDB_TEST_COMMON_ARGS} --show-inferior-console)
  endif()
endif()

if (NOT ${CMAKE_SYSTEM_NAME} MATCHES "Windows|Darwin")
  list(APPEND LLDB_TEST_COMMON_ARGS
    --env ARCHIVER=${CMAKE_AR} --env OBJCOPY=${CMAKE_OBJCOPY})
endif()

if (NOT "${LLDB_LIT_TOOLS_DIR}" STREQUAL "")
  if (NOT EXISTS "${LLDB_LIT_TOOLS_DIR}")
    message(WARNING "LLDB_LIT_TOOLS_DIR ${LLDB_LIT_TOOLS_DIR} does not exist.")
  endif()
endif()

if(CMAKE_HOST_APPLE)
  if(LLDB_BUILD_FRAMEWORK)
    get_target_property(framework_build_dir liblldb LIBRARY_OUTPUT_DIRECTORY)
    list(APPEND LLDB_TEST_COMMON_ARGS --framework ${framework_build_dir}/LLDB.framework)
  endif()

  # Use the same identity for testing
  get_property(code_sign_identity_used GLOBAL PROPERTY LLDB_DEBUGSERVER_CODESIGN_IDENTITY)
  if(code_sign_identity_used)
    list(APPEND LLDB_TEST_COMMON_ARGS --codesign-identity "${code_sign_identity_used}")
  endif()

  if(LLDB_USE_SYSTEM_DEBUGSERVER)
    lldb_find_system_debugserver(system_debugserver_path)
    add_custom_target(debugserver
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${system_debugserver_path} ${LLVM_RUNTIME_OUTPUT_INTDIR}
      COMMENT "Copying the system debugserver to LLDB's binaries directory for testing.")
    # The custom target for the system debugserver has no install target, so we
    # need to remove it from the LLVM_DISTRIBUTION_COMPONENTS list.
    if (LLVM_DISTRIBUTION_COMPONENTS)
      list(REMOVE_ITEM LLVM_DISTRIBUTION_COMPONENTS debugserver)
      set(LLVM_DISTRIBUTION_COMPONENTS ${LLVM_DISTRIBUTION_COMPONENTS} PARENT_SCOPE)
    endif()
    message(STATUS "LLDB tests use out-of-tree debugserver: ${system_debugserver_path}")
    list(APPEND LLDB_TEST_COMMON_ARGS --out-of-tree-debugserver)
    add_lldb_test_dependency(debugserver)
  elseif(TARGET debugserver)
    set(debugserver_path ${LLVM_RUNTIME_OUTPUT_INTDIR}/debugserver)
    message(STATUS "LLDB Tests use just-built debugserver: ${debugserver_path}")
    list(APPEND LLDB_TEST_COMMON_ARGS --server ${debugserver_path})
    add_lldb_test_dependency(debugserver)
  elseif(TARGET lldb-server)
    set(lldb_server_path ${LLVM_RUNTIME_OUTPUT_INTDIR}/lldb-server)
    message(STATUS "LLDB Tests use just-built lldb-server: ${lldb_server_path}")
    list(APPEND LLDB_TEST_COMMON_ARGS --server ${lldb_server_path})
    add_lldb_test_dependency(lldb-server)
  else()
    message(WARNING "LLDB Tests enabled, but no server available")
  endif()
endif()

set(LLDB_DOTEST_ARGS ${LLDB_TEST_COMMON_ARGS};${LLDB_TEST_USER_ARGS})
set_property(GLOBAL PROPERTY LLDB_DOTEST_ARGS_PROPERTY ${LLDB_DOTEST_ARGS})

add_custom_target(check-lldb)
add_dependencies(check-lldb lldb-test-deps)
set_target_properties(check-lldb PROPERTIES FOLDER "lldb misc")

# LLD is required to link test executables on Windows.
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  if (TARGET lld)
    add_lldb_test_dependency(lld)
  else ()
    message(WARNING "lld required to test LLDB on Windows")
  endif ()
endif ()

# Run the lit-style tests and the unittests as part of the check-lldb target.
add_dependencies(check-lldb check-lldb-lit)

# This will add LLDB's test dependencies to the dependencies for check-all and
# include them in the test-depends target.
set_property(GLOBAL APPEND PROPERTY LLVM_LIT_DEPENDS ${ARG_DEPENDS})
