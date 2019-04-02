set(SYCL_TESTS_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(RT_TEST_ARGS ${RT_TEST_ARGS} "-v")

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg
  )

add_lit_testsuite(check-sycl "Running the SYCL regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${RT_TEST_ARGS}
  DEPENDS sycl-toolchain FileCheck get_device_count_by_type
  )
set_target_properties(check-sycl PROPERTIES FOLDER "SYCL tests")
