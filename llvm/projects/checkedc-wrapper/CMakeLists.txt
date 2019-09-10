# This script adds the Checked C tests to the LLVM tests infrastructure,
# provided that the Checked C repo exists as a subdirectory.

# LLVM uses the CMake meta-build system to generate the build system.
# The build system includes targets for running tests.
#
# The script adds a new target for running Checked C tests.
# To do that, it sets up  the information needed by the LIT tool.

if(CHECKEDC_IN_TREE)
  add_subdirectory(checkedc/include)

  set(CHECKEDC_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
  set(CHECKEDC_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})


  if (CMAKE_CFG_INTDIR STREQUAL ".")
    set(LLVM_BUILD_MODE ".")
  else ()
    set(LLVM_BUILD_MODE "%(build_mode)s")
  endif ()

  string(REPLACE ${CMAKE_CFG_INTDIR} ${LLVM_BUILD_MODE} CLANG_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

  configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )

  list(APPEND CHECKEDC_TEST_DEPS
    clang clang-headers checkedc-headers
    )

  if(CLANG_ENABLE_STATIC_ANALYZER)
    list(APPEND CLANG_TEST_DEPS
      clang-check
      )
  endif()

  set(CHECKEDC_TEST_PARAMS
    checkedc_site_config=${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
    )

if( NOT CLANG_BUILT_STANDALONE )
  list(APPEND CHECKEDC_TEST_DEPS
    llvm-config
    FileCheck count not
    opt
    )
endif()

  add_custom_target(checkedc-test-depends DEPENDS ${CHECKEDC_TEST_DEPS})

  add_lit_testsuite(check-checkedc "Running the Checked C regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    #LIT ${LLVM_LIT}
    PARAMS ${CHECKEDC_TEST_PARAMS}
    DEPENDS ${CHECKEDC_TEST_DEPS}
    ARGS ${CHECKEDC_TEST_EXTRA_ARGS}
    )

  set_target_properties(check-checkedc PROPERTIES FOLDER "Checked C tests")

endif()
