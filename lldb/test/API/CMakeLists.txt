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

# Set the path to the default lldb test executable.
set(LLDB_DEFAULT_TEST_EXECUTABLE "${LLVM_RUNTIME_OUTPUT_INTDIR}/lldb${CMAKE_EXECUTABLE_SUFFIX}")

# Set the paths to default llvm tools.
set(LLDB_DEFAULT_TEST_DSYMUTIL "${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/dsymutil${CMAKE_EXECUTABLE_SUFFIX}")
set(LLDB_DEFAULT_TEST_FILECHECK "${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/FileCheck${CMAKE_EXECUTABLE_SUFFIX}")

if (TARGET clang)
  set(LLDB_DEFAULT_TEST_COMPILER "${LLVM_BINARY_DIR}/${CMAKE_CFG_INTDIR}/bin/clang${CMAKE_EXECUTABLE_SUFFIX}")
else()
  set(LLDB_DEFAULT_TEST_COMPILER "")
endif()

set(LLDB_TEST_EXECUTABLE "${LLDB_DEFAULT_TEST_EXECUTABLE}" CACHE PATH "lldb executable used for testing")
set(LLDB_TEST_COMPILER "${LLDB_DEFAULT_TEST_COMPILER}" CACHE PATH "C Compiler to use for building LLDB test inferiors")
set(LLDB_TEST_DSYMUTIL "${LLDB_DEFAULT_TEST_DSYMUTIL}" CACHE PATH "dsymutil used for generating dSYM bundles")
set(LLDB_TEST_FILECHECK "${LLDB_DEFAULT_TEST_FILECHECK}" CACHE PATH "FileCheck used for testing purposes")

if ("${LLDB_TEST_COMPILER}" STREQUAL "")
  message(FATAL_ERROR "LLDB test compiler not specified. Tests will not run.")
endif()

list(APPEND LLDB_TEST_COMMON_ARGS
  --executable ${LLDB_TEST_EXECUTABLE}
  --compiler ${LLDB_TEST_COMPILER}
  --dsymutil ${LLDB_TEST_DSYMUTIL}
  --filecheck ${LLDB_TEST_FILECHECK}
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
    list(APPEND LLDB_TEST_COMMON_ARGS --framework ${LLDB_FRAMEWORK_ABSOLUTE_BUILD_DIR}/LLDB.framework)
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

set(LLDB_DOTEST_ARGS ${LLDB_TEST_COMMON_ARGS};${LLDB_TEST_USER_ARGS} CACHE INTERNAL STRING)
set(dotest_args_replacement ${LLVM_BUILD_MODE})

if(LLDB_BUILT_STANDALONE)
  # In paths to our build-tree, replace CMAKE_CFG_INTDIR with our configuration name placeholder.
  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} config_runtime_output_dir ${LLVM_RUNTIME_OUTPUT_INTDIR})
  string(REPLACE ${LLVM_RUNTIME_OUTPUT_INTDIR} ${config_runtime_output_dir} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")

  # Remaining ones must be paths to the provided LLVM build-tree.
  if(LLVM_CONFIGURATION_TYPES)
    # LLDB uses single-config; LLVM multi-config; pick one and prefer Release types.
    # Otherwise, if both use multi-config the default is fine.
    if(NOT CMAKE_CONFIGURATION_TYPES)
      if(RelWithDebInfo IN_LIST LLVM_CONFIGURATION_TYPES)
        set(dotest_args_replacement RelWithDebInfo)
      elseif(Release IN_LIST LLVM_CONFIGURATION_TYPES)
        set(dotest_args_replacement Release)
      else()
        list(GET LLVM_CONFIGURATION_TYPES 0 dotest_args_replacement)
      endif()
    endif()
  else()
    # Common case: LLVM used a single-configuration generator like Ninja.
    set(dotest_args_replacement ".")
  endif()
endif()

string(REPLACE ${CMAKE_CFG_INTDIR} ${dotest_args_replacement} LLDB_DOTEST_ARGS "${LLDB_DOTEST_ARGS}")

# Configure the API test suite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
