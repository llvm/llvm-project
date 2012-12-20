set(POLLY_TEST_DIRECTORIES
  "AffineIterator"
  "ScopInfo"
  "ScheduleOptimizer"
  "CodeGen"
  "Cloog"
  "OpenMP"
  "polybench"
  "vect")


set(POLLY_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")
set(POLLY_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/..")
set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")

if (NOT DEFINED LLVM_MAIN_SRC_DIR)

  # We are building polly out of tree, adjust the settings.
  # FIXME: FileCheck is not available in llvm install directory at the moment.
  set(LLVM_LIT ${LLVM_INSTALL_ROOT}/bin/llvm-lit)
  set(POLLY_TEST_DEPS LLVMPolly)

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

    foreach(testdir ${POLLY_TEST_DIRECTORIES})
      add_custom_target(polly-test-${testdir}
        COMMAND ${LLVM_LIT}
                    --param polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                    --param build_config=${CMAKE_CFG_INTDIR}
                    -sv ${POLLY_TEST_EXTRA_ARGS}
                    ${CMAKE_CURRENT_BINARY_DIR}/${testdir}
                    DEPENDS ${POLLY_TEST_DEPS}
                    COMMENT "Running Polly regression tests in ${testdir}")
    endforeach()

    add_custom_target(check-polly
      COMMAND ${LLVM_LIT}
                  --param polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
                  --param build_config=${CMAKE_CFG_INTDIR}
                  -sv ${POLLY_TEST_EXTRA_ARGS}
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS ${POLLY_TEST_DEPS}
                  COMMENT "Running Polly regression tests")
  endif()

else (NOT DEFINED LLVM_MAIN_SRC_DIR)

  set(LLVM_LIT ${LLVM_TOOLS_BINARY_DIR}/llvm-lit)
  set(POLLY_TEST_DEPS opt lli LLVMPolly FileCheck not)

  set(LLVM_BINARY_DIR "${LLVM_BINARY_DIR}")
  set(LLVM_TOOLS_DIR "${LLVM_TOOLS_BINARY_DIR}")
  set(LLVM_LIBS_DIR "${LLVM_BINARY_DIR}/lib")
  set(POLLY_LIB_DIR "${LLVM_LIBS_DIR}")

  add_lit_testsuite(check-polly "Running polly regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    PARAMS polly_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    DEPENDS ${POLLY_TEST_DEPS}
    )

  set_target_properties(check-polly PROPERTIES FOLDER "polly tests")

endif (NOT DEFINED LLVM_MAIN_SRC_DIR)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg)

# Add a legacy target spelling: polly-test
add_custom_target(polly-test)
add_dependencies(polly-test check-polly)
