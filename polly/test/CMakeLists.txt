set(POLLY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(POLLY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")
set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")

add_custom_target(check-polly)
set_target_properties(check-polly PROPERTIES FOLDER "Polly")

if (NOT DEFINED LLVM_MAIN_SRC_DIR)

  # We are building polly out of tree, adjust the settings.
  # FIXME: FileCheck is not available in llvm install directory at the moment.
  set(LLVM_LIT ${LLVM_INSTALL_ROOT}/bin/llvm-lit)
  if (POLLY_BUNDLED_ISL)
    set(POLLY_TEST_DEPS LLVMPolly polly-isl-test)
  endif (POLLY_BUNDLED_ISL)
  if (POLLY_GTEST_AVAIL)
    list(APPEND POLLY_TEST_DEPS PollyUnitTests)
  endif ()

  set(LLVM_BINARY_DIR "${LLVM_INSTALL_ROOT}")
  set(LLVM_TOOLS_DIR "${LLVM_INSTALL_ROOT}/bin")
  set(LLVM_LIBS_DIR "${LLVM_INSTALL_ROOT}/lib")
  set(POLLY_LIB_DIR "${POLLY_BINARY_DIR}/lib")

  include(FindPythonInterp)
  if(PYTHONINTERP_FOUND)
    option(POLLY_TEST_DISABLE_BAR "Run Polly tests with --no-progress-bar" OFF)
    set(POLLY_TEST_EXTRA_ARGS)
    if (MSVC OR XCODE OR POLLY_TEST_DISABLE_BAR)
      set(POLLY_TEST_EXTRA_ARGS "--no-progress-bar")
    endif()

    option(POLLY_TEST_USE_VG "Run Polly tests under Valgrind" OFF)
    if(POLLY_TEST_USE_VG)
      set(POLLY_TEST_EXTRA_ARGS ${POLLY_TEST_EXTRA_ARGS} "--vg")
    endif ()

    # Parameters required for lit.site.cfg.in
    set(LLVM_SOURCE_DIR ${LLVM_SOURCE_ROOT})
    if (CMAKE_CFG_INTDIR STREQUAL ".")
      set(LLVM_BUILD_MODE ".")
    else ()
      set(LLVM_BUILD_MODE "%(build_mode)s")
    endif ()
    set(ENABLE_SHARED "1")
    set(SHLIBDIR "${LLVM_BINARY_DIR}/bin")
    set(LINK_POLLY_INTO_TOOLS "OFF")

    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

    add_custom_target(check-polly-tests
      COMMAND ${LLVM_LIT}
                  --param polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                  --param polly_unit_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
                  --param build_config=${CMAKE_CFG_INTDIR}
                  -sv ${POLLY_TEST_EXTRA_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${POLLY_TEST_DEPS}
                  COMMENT "Running Polly regression/unit tests")
    set_target_properties(check-polly-tests PROPERTIES FOLDER "Polly")
    add_dependencies(check-polly check-polly-tests)

    if (POLLY_GTEST_AVAIL)
      configure_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
        ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg)

      add_custom_target(check-polly-unittests
        COMMAND ${LLVM_LIT}
                  --param polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
                  --param build_config=${CMAKE_CFG_INTDIR}
                  -sv ${POLLY_TEST_EXTRA_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}/Unit
                  DEPENDS PollyUnitTests
                  COMMENT "Running Polly unit tests")
      set_target_properties(check-polly-unittests PROPERTIES FOLDER "Polly")
    endif ()
    configure_file(
      ${CMAKE_CURRENT_SOURCE_DIR}/UnitIsl/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/UnitIsl/lit.site.cfg)


    if (POLLY_BUNDLED_ISL)
      add_custom_target(check-polly-isl
        command ${LLVM_LIT}
                  --param polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/UnitIsl/lit.site.cfg
                  --param build_config=${CMAKE_CFG_INTDIR}
                  -sv ${POLLY_TEST_EXTRA_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS polly-isl-test
                  COMMENT "Running isl unit tests")
      set_target_properties(check-polly-isl PROPERTIES FOLDER "Polly")
    endif (POLLY_BUNDLED_ISL)
  endif()

else (NOT DEFINED LLVM_MAIN_SRC_DIR)

  set(LLVM_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
  set(POLLY_TEST_DEPS llvm-config opt LLVMPolly FileCheck not)
  if (POLLY_BUNDLED_ISL)
    list(APPEND POLLY_TEST_DEPS polly-isl-test)
  endif()
  if (POLLY_GTEST_AVAIL)
    list(APPEND POLLY_TEST_DEPS PollyUnitTests)
  endif ()

  set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
  set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
  set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib${LLVM_LIBDIR_SUFFIX}")
  set(POLLY_LIB_DIR "${LLVM_LIBS_DIR}")

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

  if (POLLY_GTEST_AVAIL)
    configure_lit_site_cfg(
      ${CMAKE_CURRENT_SOURCE_DIR}/Unit/lit.site.cfg.in
      ${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
    )

    # Run only unit tests
    add_lit_testsuite(check-polly-unittests "Running polly unit tests only"
      ${CMAKE_CURRENT_BINARY_DIR}/Unit
      PARAMS polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/Unit/lit.site.cfg
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

endif (NOT DEFINED LLVM_MAIN_SRC_DIR)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/update_check.py
  ${CMAKE_CURRENT_BINARY_DIR}/update_check.py)
file(COPY ${CMAKE_CURRENT_BINARY_DIR}/update_check.py
  DESTINATION ${LLVM_TOOLS_DIR}
  FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ
  GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)

# Add a legacy target spelling: polly-test
add_custom_target(polly-test)
set_target_properties(polly-test PROPERTIES FOLDER "Polly")
add_dependencies(polly-test check-polly)
