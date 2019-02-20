function(add_python_test_target name test_script args comment)
  set(PYTHON_TEST_COMMAND
    ${PYTHON_EXECUTABLE}
    ${test_script}
    ${args}
    )

  add_custom_target(${name}
    COMMAND ${PYTHON_TEST_COMMAND} ${ARG_DEFAULT_ARGS}
    COMMENT "${comment}"
    DEPENDS ${LLDB_TEST_DEPS}
    USES_TERMINAL
    )
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

# BEGIN - Swift Mods
option(LLDB_TEST_SWIFT "Use in-tree swift when testing lldb" On)

if(LLDB_TEST_SWIFT)
  set(SWIFT_TEST_ARGS
    --swift-compiler ${LLDB_SWIFTC}
    --swift-library ${LLDB_SWIFT_LIBS})
endif()
# END - Swift Mods

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
  --build-dir
  ${CMAKE_BINARY_DIR}/lldb-test-build.noindex
  -S nm
  -u CXXFLAGS
  -u CFLAGS
  )

list(APPEND LLDB_TEST_COMMON_ARGS
  --executable ${LLDB_TEST_EXECUTABLE}
  --dsymutil ${LLDB_TEST_DSYMUTIL}
  --filecheck ${LLDB_TEST_FILECHECK}
  -C ${LLDB_TEST_C_COMPILER}
  ${SWIFT_TEST_ARGS}
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

if(LLDB_CODESIGN_IDENTITY_USED)
  list(APPEND LLDB_TEST_COMMON_ARGS --codesign-identity "${LLDB_CODESIGN_IDENTITY_USED}")
endif()

if(LLDB_BUILD_FRAMEWORK)
  get_target_property(framework_target_dir liblldb LIBRARY_OUTPUT_DIRECTORY)
  list(APPEND LLDB_TEST_COMMON_ARGS --framework ${framework_target_dir}/LLDB.framework)
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

if(CMAKE_HOST_APPLE AND DEBUGSERVER_PATH)
  list(APPEND LLDB_TEST_COMMON_ARGS --server ${DEBUGSERVER_PATH})
endif()

if(SKIP_TEST_DEBUGSERVER)
  list(APPEND LLDB_TEST_COMMON_ARGS --out-of-tree-debugserver)
endif()

set(LLDB_DOTEST_ARGS ${LLDB_TEST_COMMON_ARGS};${LLDB_TEST_USER_ARGS})
set_property(GLOBAL PROPERTY LLDB_DOTEST_ARGS_PROPERTY ${LLDB_DOTEST_ARGS})

add_python_test_target(check-lldb-single
  ${LLDB_SOURCE_DIR}/test/dotest.py
  "--no-multiprocess;${LLDB_DOTEST_ARGS}"
  "Testing LLDB with args: ${LLDB_DOTEST_ARGS}"
  )

# If tests crash cause LLDB to crash, or things are otherwise unstable, or if machine-parsable
# output is desired (i.e. in continuous integration contexts) check-lldb-single is a better target.
add_custom_target(check-lldb)

# Make check-lldb depend on all test dependencies.
add_dependencies(check-lldb lldb-test-deps)

# If we're building with an in-tree clang, then list clang as a dependency
# to run tests.
if (TARGET clang)
  add_dependencies(check-lldb clang)
endif()

# If we build the framework, tests will use the tools in there.
if (TARGET lldb-framework)
  add_dependencies(check-lldb lldb-framework)
endif()

# LLD is required to link test executables on Windows.
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  if (TARGET lld)
    add_dependencies(check-lldb lld)
  else ()
    message(WARNING "lld required to test LLDB on Windows")
  endif ()
endif ()

# Run the lit-style tests and the unittests as part of the check-lldb target.
add_dependencies(check-lldb check-lldb-lit)

# This will add LLDB's test dependencies to the dependencies for check-all and
# include them in the test-depends target.
set_property(GLOBAL APPEND PROPERTY LLVM_LIT_DEPENDS ${ARG_DEPENDS})
