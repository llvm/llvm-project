configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

list(APPEND ORC_RT_TEST_DEPS
  FileCheck
  count
  not
  orc-executor
  )

add_custom_target(orc-rt-test-depends DEPENDS ${ORC_RT_TEST_DEPS})
set_target_properties(orc-rt-test-depends PROPERTIES FOLDER "orc-rt/Tests")

add_lit_testsuite(check-orc-rt "Running the ORCRT regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${ORC_RT_TEST_DEPS}
  )

add_lit_testsuites(ORC-RT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${ORC_RT_TEST_DEPS}
)

# Add orc-rt unit tests
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/unit/lit.site.cfg
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/unit/lit.cfg.py)

add_lit_testsuite(check-orcrt-unit "Running orc-rt unittest suites"
  ${CMAKE_CURRENT_BINARY_DIR}/unit
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS OrcRTUnitTests)

