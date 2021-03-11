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
add_custom_target(lldb-test-deps)
set_target_properties(lldb-test-deps PROPERTIES FOLDER "lldb misc")

function(add_lldb_test_dependency)
  foreach(dependency ${ARGN})
    add_dependencies(lldb-test-deps ${dependency})
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

if(TARGET lldb-instr)
  add_lldb_test_dependency(lldb-instr)
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

  if(APPLE)
    # FIXME: Standalone builds should import the cxx target as well.
    if(LLDB_BUILT_STANDALONE)
      # For now check that the include directory exists.
      set(cxx_dir "${LLVM_BINARY_DIR}/include/c++")
      if(NOT EXISTS ${cxx_dir})
        message(WARNING "LLDB test suite requires libc++ in llvm/projects/libcxx or an existing build symlinked to ${cxx_dir}")
      endif()
    else()
      # We require libcxx for the test suite, so if we aren't building it,
      # try to provide a helpful error about how to resolve the situation.
      if(NOT TARGET cxx AND NOT libcxx IN_LIST LLVM_ENABLE_RUNTIMES)
        if(LLVM_ENABLE_PROJECTS STREQUAL "")
          # If `LLVM_ENABLE_PROJECTS` is not being used (implying that we are
          # using the old layout), suggest checking it out.
          message(FATAL_ERROR
            "LLDB test suite requires libc++, but it is currently disabled. "
            "Please checkout `libcxx` in `llvm/projects` or disable tests "
            "via `LLDB_INCLUDE_TESTS=OFF`.")
        else()
          # If `LLVM_ENABLE_PROJECTS` is being used, suggest adding it.
          message(FATAL_ERROR
            "LLDB test suite requires libc++, but it is currently disabled. "
            "Please add `libcxx` to `LLVM_ENABLE_PROJECTS` or "
            "`LLVM_ENABLE_RUNTIMES`, or disable tests via "
            "`LLDB_INCLUDE_TESTS=OFF`.")
        endif()
      endif()
      add_lldb_test_dependency(cxx)
    endif()
  endif()
endif()

if (LLDB_BUILT_STANDALONE)
  set(LLVM_HOST_TRIPLE ${TARGET_TRIPLE})
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

add_lit_testsuites(LLDB
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS lldb-test-deps)

add_lit_testsuite(check-lldb-lit "Running lldb lit test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS lldb-test-deps)
set_target_properties(check-lldb-lit PROPERTIES FOLDER "lldb tests")

add_custom_target(check-lldb)
add_dependencies(check-lldb lldb-test-deps)
set_target_properties(check-lldb PROPERTIES FOLDER "lldb misc")
add_dependencies(check-lldb check-lldb-lit)

# Add a lit test suite that runs the API & shell test while capturing a
# reproducer.
add_lit_testsuite(check-lldb-reproducers-capture
  "Running lldb test suite with reproducer capture"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  ${CMAKE_CURRENT_BINARY_DIR}/Shell
  PARAMS "lldb-run-with-repro=capture"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-deps)

# Add a lit test suite that runs the API & shell test by replaying a
# reproducer.
add_lit_testsuite(check-lldb-reproducers
  "Running lldb test suite with reproducer replay"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  ${CMAKE_CURRENT_BINARY_DIR}/Shell
  PARAMS "lldb-run-with-repro=replay"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-deps)
add_dependencies(check-lldb-reproducers check-lldb-reproducers-capture)

# Targets for running the test suite on the different Apple simulators.
add_lit_testsuite(check-lldb-simulator-ios
  "Running lldb test suite on the iOS simulator"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  PARAMS "lldb-run-with-simulator=ios"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-deps)

add_lit_testsuite(check-lldb-simulator-watchos
  "Running lldb test suite on the watchOS simulator"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  PARAMS "lldb-run-with-simulator=watchos"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-deps)

add_lit_testsuite(check-lldb-simulator-tvos
  "Running lldb test suite on the tvOS simulator"
  ${CMAKE_CURRENT_BINARY_DIR}/API
  PARAMS "lldb-run-with-simulator=tvos"
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-test-deps)


if(LLDB_BUILT_STANDALONE)
  # This has to happen *AFTER* add_lit_testsuite.
  if (EXISTS ${LLVM_MAIN_SRC_DIR}/utils/llvm-lit)
    # LLVM's make_paths_relative uses Python3_EXECUTABLE which isn't set in a
    # standalone LLDB build.
    set(Python3_EXECUTABLE ${Python3_EXECUTABLE})
    add_subdirectory(${LLVM_MAIN_SRC_DIR}/utils/llvm-lit ${CMAKE_CURRENT_BINARY_DIR}/llvm-lit)
  endif()
endif()
