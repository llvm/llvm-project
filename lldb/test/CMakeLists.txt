# Test runner infrastructure for LLDB. This configures the LLDB test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

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
      if(NOT TARGET cxx)
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
            "Please add `libcxx` to `LLVM_ENABLE_PROJECTS` or disable tests "
            "via `LLDB_INCLUDE_TESTS=OFF`.")
        endif()
      endif()
      add_lldb_test_dependency(cxx)
    endif()
  endif()
endif()

add_lldb_test_dependency(
  lit-cpuid
  llc
  lli
  llvm-config
  llvm-dwarfdump
  llvm-nm
  llvm-mc
  llvm-objcopy
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
  LLDB_ENABLE_PYTHON
  LLDB_ENABLE_LUA
  LLDB_ENABLE_LZMA
  LLVM_ENABLE_ZLIB
  LLVM_ENABLE_SHARED_LIBS
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
