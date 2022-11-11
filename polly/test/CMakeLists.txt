set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")

add_custom_target(check-polly)
set_target_properties(check-polly PROPERTIES FOLDER "Polly")

if(NOT LLVM_MAIN_SRC_DIR)
  find_program(LLVM_OPT NAMES opt HINTS ${LLVM_TOOLS_BINARY_DIR})
  find_program(LLVM_FILECHECK NAMES FileCheck HINTS ${LLVM_TOOLS_BINARY_DIR})
  find_program(LLVM_NOT NAMES not HINTS ${LLVM_TOOLS_BINARY_DIR})
  if (NOT LLVM_OPT)
    message(WARNING "LLVM's opt program could not be found. Please set LLVM_OPT.")
  endif()
  if (NOT LLVM_FILECHECK)
    message(WARNING "LLVM's FileCheck program could not be found. "
      "Please set LLVM_FILECHECK. Please set LLVM_FILECHECK.")
  endif()
  if (NOT LLVM_NOT)
    message(WARNING "LLVM's not program could not be found. Please set LLVM_NOT.")
  endif()
  get_filename_component(EXTRA_PATHS ${LLVM_OPT} DIRECTORY)
  list(APPEND POLLY_TEST_EXTRA_PATHS "${EXTRA_PATHS}")
  get_filename_component(EXTRA_PATHS ${LLVM_FILECHECK} DIRECTORY)
  list(APPEND POLLY_TEST_EXTRA_PATHS "${EXTRA_PATHS}")
  get_filename_component(EXTRA_PATHS ${LLVM_NOT} DIRECTORY)
  list(APPEND POLLY_TEST_EXTRA_PATHS "${EXTRA_PATHS}")
  list(REMOVE_DUPLICATES POLLY_TEST_EXTRA_PATHS)
  message(STATUS "Extra paths: ${POLLY_TEST_EXTRA_PATHS}")
  if ("${POLLY_TEST_EXTRA_PATHS}" STREQUAL "${LLVM_TOOLS_BINARY_DIR}")
    set(POLLY_TEST_EXTRA_PATHS "")
  endif()
  set(POLLY_TEST_DEPS LLVMPolly)
else ()
  set(LLVM_OPT "${LLVM_TOOLS_BINARY_DIR}/opt")
  set(LLVM_FILECHECK "${LLVM_TOOLS_BINARY_DIR}/FileCheck")
  set(LLVM_NOT "${LLVM_TOOLS_BINARY_DIR}/not")
  set(POLLY_TEST_EXTRA_PATHS "")
  set(POLLY_TEST_DEPS llvm-config opt LLVMPolly FileCheck not count)
endif()

if (POLLY_BUNDLED_ISL)
  list(APPEND POLLY_TEST_DEPS polly-isl-test)
endif()
if (POLLY_GTEST_AVAIL)
  list(APPEND POLLY_TEST_DEPS PollyUnitTests)
endif ()

set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
set(LLVM_LIBS_DIR "${LLVM_LIBRARY_DIR}")
if (CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  set(POLLY_LIB_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
else()
  set(POLLY_LIB_DIR "${POLLY_BINARY_DIR}/lib")
endif()

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

# Run regression and unit tests
add_lit_testsuite(check-polly-tests "Running polly regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  PARAMS polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  polly_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  DEPENDS ${POLLY_TEST_DEPS}
  )
set_target_properties(check-polly-tests PROPERTIES FOLDER "Polly")
add_dependencies(check-polly check-polly-tests)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
  )

if (POLLY_GTEST_AVAIL)
  # Run only unit tests
  add_lit_testsuite(check-polly-unittests "Running polly unit tests only"
    ${CMAKE_CURRENT_BINARY_DIR}/Unit
    PARAMS polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS PollyUnitTests
    )
  set_target_properties(check-polly-unittests PROPERTIES FOLDER "Polly")
endif ()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/UnitIsl/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/UnitIsl/lit.site.cfg)

if (POLLY_BUNDLED_ISL)
  add_lit_testsuite(check-polly-isl "Running isl unit tests only"
    ${CMAKE_CURRENT_BINARY_DIR}/UnitIsl
    PARAMS polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/UnitIsl/lit.site.cfg
    EXCLUDE_FROM_CHECK_ALL
    DEPENDS polly-isl-test
    )
  set_target_properties(check-polly-isl PROPERTIES FOLDER "Polly")
endif (POLLY_BUNDLED_ISL)

# Run polly-check-format as part of polly-check only if we are compiling with
# clang, so clang-format is available.
# if (TARGET clang-format) would be preferable, but this target is only added
# after Polly, i.e. there is no such target yet at this point. The CMake cache
# entry LLVM_TOOL_CLANG_BUILD will only be defined after clang has been
# configured at least once, i.e. format will be checked only after a rerun of
# CMake's configure.
if (LLVM_TOOL_CLANG_BUILD)
  add_dependencies(check-polly polly-check-format)
endif ()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/update_check.py
  ${CMAKE_CURRENT_BINARY_DIR}/update_check.py)

# Add a legacy target spelling: polly-test
add_custom_target(polly-test)
set_target_properties(polly-test PROPERTIES FOLDER "Polly")
add_dependencies(polly-test check-polly)
