add_custom_target(lldb-unit-test-deps)
set_target_properties(lldb-unit-test-deps PROPERTIES FOLDER "LLDB/Tests")
add_dependencies(lldb-unit-test-deps lldb-test-depends)

add_lit_testsuites(LLDB-UNIT
  ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS lldb-unit-test-deps)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

add_lit_testsuite(check-lldb-unit "Running lldb unit test suite"
  ${CMAKE_CURRENT_BINARY_DIR}
  EXCLUDE_FROM_CHECK_ALL
  DEPENDS lldb-unit-test-deps)
