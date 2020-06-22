# Configure the Shell test suite.
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit-lldb-init.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit-lldb-init)

if (CMAKE_GENERATOR STREQUAL "Xcode")
  # Xcode does not get the auto-generated targets. We need to create
  # check-lldb-shell manually.
  add_lit_testsuite(check-lldb-shell "Running lldb shell test suite"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS lldb-test-deps)
endif()
