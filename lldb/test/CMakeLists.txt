# Test runner infrastructure for LLDB. This configures the LLDB test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

if(LLDB_BUILT_STANDALONE)
  # In order to run check-lldb-* we need the correct map_config directives in
  # llvm-lit. Because this is a standalone build, LLVM doesn't know about LLDB,
  # and the lldb mappings are missing. We build our own llvm-lit, and tell LLVM
  # to use the llvm-lit in the lldb build directory.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/llvm-lit)
    set(LLVM_EXTERNAL_LIT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llvm-lit)
  endif()
endif()

# Configure the build directory.
# The .noindex suffix is a marker for Spotlight to never index the
# build directory.  LLDB queries Spotlight to locate .dSYM bundles
# based on the UUID embedded in a binary, and because the UUID is a
# hash of filename and .text section, there *will* be conflicts inside
# the build directory.
set(LLDB_TEST_BUILD_DIRECTORY "${PROJECT_BINARY_DIR}/lldb-test-build.noindex" CACHE PATH "The build root for building tests.")

# Configure and create module cache directories.
set(LLDB_TEST_MODULE_CACHE_LLDB "${LLDB_TEST_BUILD_DIRECTORY}/module-cache-lldb" CACHE PATH "The Clang module cache used by the Clang embedded in LLDB while running tests.")
set(LLDB_TEST_MODULE_CACHE_CLANG "${LLDB_TEST_BUILD_DIRECTORY}/module-cache-clang" CACHE PATH "The Clang module cache used by the Clang while building tests.")
file(MAKE_DIRECTORY ${LLDB_TEST_MODULE_CACHE_LLDB})
file(MAKE_DIRECTORY ${LLDB_TEST_MODULE_CACHE_CLANG})

# LLVM_BUILD_MODE is used in lit.site.cfg
if (CMAKE_CFG_INTDIR STREQUAL ".")
  set(LLVM_BUILD_MODE ".")
else ()
  set(LLVM_BUILD_MODE "%(build_mode)s")
endif ()

string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_LIBS_DIR ${LLVM_LIBRARY_OUTPUT_INTDIR})
string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} LLDB_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

# Create a custom target to track test dependencies.
add_custom_target(lldb-test-depends)
set_target_properties(lldb-test-depends PROPERTIES FOLDER "lldb misc")

# Create an alias for the legacy name of lldb-test-depends
add_custom_target(lldb-test-deps)
add_dependencies(lldb-test-deps lldb-test-depends)

function(add_lldb_test_dependency)
  foreach(dependency ${ARGN})
    add_dependencies(lldb-test-depends ${dependency})
  endforeach()
endfunction(add_lldb_test_dependency)

# lldb itself and lldb-test is an hard dependency for the testsuites.
add_lldb_test_dependency(lldb)
add_lldb_test_dependency(lldb-test)

# On Darwin, darwin-debug is an hard dependency for the testsuites.
if (CMAKE_SYSTEM_NAME MATCHES "Darwin")
  add_lldb_test_dependency(darwin-debug)
endif()

if(TARGET debugserver)
  add_lldb_test_dependency(debugserver)
endif()
if(TARGET lldb-server)
  add_lldb_test_dependency(lldb-server)
endif()

if(TARGET lldb-vscode)
  add_lldb_test_dependency(lldb-vscode)
endif()

if(TARGET liblldb)
  add_lldb_test_dependency(liblldb)
endif()

if(TARGET lldb-framework)
  add_lldb_test_dependency(lldb-framework)
endif()

# Add dependencies that are not exported targets when building standalone.
if(NOT LLDB_BUILT_STANDALONE)
  add_lldb_test_dependency(
    FileCheck
    count
    dsymutil
    llvm-strip
    not
    split-file
    yaml2obj
  )
endif()

# Add dependencies if we test with the in-tree clang.
# This works with standalone builds as they import the clang target.
if(TARGET clang)
  add_lldb_test_dependency(clang)

  if(TARGET asan)
    add_lldb_test_dependency(asan)
  endif()

  if(TARGET tsan)
    add_lldb_test_dependency(tsan)
  endif()

  if (TARGET compiler-rt)
    add_lldb_test_dependency(compiler-rt)
  endif()

  if(APPLE AND NOT LLVM_TARGET_IS_CROSSCOMPILE_HOST)
    # FIXME: Standalone builds should import the cxx target as well.
    if(LLDB_BUILT_STANDALONE)
      # For now check that the include directory exists.
      set(cxx_dir "${LLVM_BINARY_DIR}/include/c++")
      if(NOT EXISTS ${cxx_dir})
        message(WARNING "LLDB test suite requires libc++ in llvm/projects/libcxx or an existing build symlinked to ${cxx_dir}")
      endif()
    else()
      # We require libcxx for the test suite, so if we aren't building it,
      # provide a helpful error about how to resolve the situation.
      if(NOT TARGET cxx AND NOT libcxx IN_LIST LLVM_ENABLE_RUNTIMES)
        message(FATAL_ERROR
          "LLDB test suite requires libc++, but it is currently disabled. "
          "Please add `libcxx` to `LLVM_ENABLE_RUNTIMES` or disable tests via "
          "`LLDB_INCLUDE_TESTS=OFF`.")
      endif()
    endif()
  endif()

  # Add libc++ dependency for libc++-specific tests. This is an optional
  # dependency as it's also possible to run the libc++ tests against the libc++
  # installed on the system.
  if (TARGET cxx)
    set(LLDB_HAS_LIBCXX ON)
    add_lldb_test_dependency(cxx)
  endif()

endif()

if (LLDB_BUILT_STANDALONE)
  set(LLVM_HOST_TRIPLE ${LLVM_TARGET_TRIPLE})
endif()

add_lldb_test_dependency(
  lit-cpuid
  llc
  lli
  llvm-config
  llvm-dwarfdump
  llvm-dwp
  llvm-nm
  llvm-mc
  llvm-objcopy
  llvm-pdbutil
  llvm-readobj
  llvm-ar
  )

if(TARGET lld)
  add_lldb_test_dependency(lld)
else()
  # LLD is required to link test executables on Windows.
  if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    message(WARNING "lld required to test LLDB on Windows")
  endif()
endif()

if (CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(LLDB_IS_64_BITS 1)
endif()

# These values are not canonicalized within LLVM.
llvm_canonicalize_cmake_booleans(
  LLDB_BUILD_INTEL_PT
  LLDB_ENABLE_PYTHON
  LLDB_ENABLE_LUA
  LLDB_ENABLE_LZMA
  LLVM_ENABLE_ZLIB
  LLVM_ENABLE_SHARED_LIBS
  LLDB_HAS_LIBCXX
  LLDB_USE_SYSTEM_DEBUGSERVER
  LLDB_IS_64_BITS)

# Configure the individual test suites.
add_subdirectory(API)
add_subdirectory(Shell)
add_subdirectory(Unit)

# Configure the top level test suite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

add_lit_testsuite(check-lldb "Running lldb lit test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS
    lldb-api-test-deps
    lldb-shell-test-deps
    lldb-unit-test-deps)
set_target_properties(check-lldb PROPERTIES FOLDER "lldb tests")

# Add a lit test suite that runs the API & shell test while capturing a
# reproducer.
add_lit_testsuite(check-lldb-reproducers-capture
  "Running lldb test suite with reproducer capture"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  ${CMAKE_CURRENT_BINARY_DIR}/Shell
  PARAMS "lldb-run-with-repro=capture"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-depends)

# Add a lit test suite that runs the API & shell test by replaying a
# reproducer.
add_lit_testsuite(check-lldb-reproducers
  "Running lldb test suite with reproducer replay"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  ${CMAKE_CURRENT_BINARY_DIR}/Shell
  PARAMS "lldb-run-with-repro=replay"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-depends)
add_dependencies(check-lldb-reproducers check-lldb-reproducers-capture)

if(LLDB_BUILT_STANDALONE)
  # This has to happen *AFTER* add_lit_testsuite.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/llvm-lit)
    # LLVM's make_paths_relative uses Python3_EXECUTABLE which isn't set in a
    # standalone LLDB build.
    set(Python3_EXECUTABLE ${Python3_EXECUTABLE})
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit ${CMAKE_CURRENT_BINARY_DIR}/llvm-lit)
  endif()
endif()
