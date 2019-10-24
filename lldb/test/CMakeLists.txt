# Test runner infrastructure for LLDB. This configures the LLDB test trees
# for use by Lit, and delegates to LLVM's lit test handlers.

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

if(NOT LLDB_BUILT_STANDALONE)
  # Since llvm-strip is a symlink created by add_custom_target, it doesn't
  # expose an export target when building standalone.
  add_lldb_test_dependency(
    llvm-strip
    FileCheck
    count
    not
  )
endif()

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
  LLDB_DISABLE_PYTHON
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
